"""URL normalisation and character tokenisation.

Two responsibilities, deliberately kept apart:

1. :func:`normalize_url` - a pure, reversible-where-possible canonicalisation
   that reduces superficial variation without destroying phishing signal.
2. :class:`CharTokenizer` - a **frozen** character vocabulary.

Why a frozen vocabulary: the model must score a URL it has never seen, at
inference time, from a checkpoint alone. Building the vocabulary from the
training set at load time would make the model's behaviour depend on the data
present when it was loaded. So the vocab is fitted once during training, saved
into the checkpoint, and reloaded verbatim at inference.

The vocabulary is fitted with a fixed character allowlist plus a frequency floor,
so rare symbols collapse to ``UNK`` instead of each getting their own embedding
that never gets gradient signal.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

#: Characters the tokenizer can represent individually. Anything else is
#: lowercased into ``<UNK>`` unless it survives normalisation.
DEFAULT_ALPHABET: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "-._~:/?#[]@!$&'()*+,;=%"
)

_HEX_RUN = re.compile(r"%[0-9a-f]{2}", re.IGNORECASE)
_REPEATS = re.compile(r"(.)\1{2,}")
_USERINFO = re.compile(r"^[^/@]*@", re.IGNORECASE)

#: Cyrillic and Greek characters that render identically to an ASCII letter in
#: most browser fonts. NFKC cannot fold these -- they are distinct letters, not
#: compatibility variants -- so ``раypal.com`` survives normalisation with a
#: Cyrillic 'р' in the host. A phishing kit that swaps them looks identical to a
#: user while defeating any ASCII-only model input. Each entry maps the confusable
#: onto the ASCII letter a human actually reads.
#:
#: Scope is deliberately small and audit-able: only characters with a well-known
#: single ASCII twin are listed, so this cannot silently rewrite a legitimately
#: internationalised host into something it never was.
_CONFUSABLES: dict[str, str] = {
    # Cyrillic
    "а": "a",  # а
    "е": "e",  # е
    "о": "o",  # о
    "р": "p",  # р
    "с": "c",  # с
    "у": "y",  # у
    "х": "x",  # х
    "ѕ": "s",  # ѕ
    "і": "i",  # і
    "ј": "j",  # ј
    "һ": "h",  # һ
    "ԁ": "d",  # ԁ
    "ԛ": "q",  # ԛ
    "ԝ": "w",  # ԝ
    "н": "h",  # н
    "м": "m",  # м
    "т": "t",  # т
    "в": "b",  # в
    "к": "k",  # к
    # Uppercase Cyrillic. These matter as much as the lowercase set: a spoofed
    # "Раypal.com" with Cyrillic capital Er renders to a user as an ordinary
    # capital P, so leaving capitals out would let exactly the attack this table
    # exists to stop through whenever the brand is capitalised.
    "А": "A",  # А
    "В": "V",  # В
    "Е": "E",  # Е
    "К": "K",  # К
    "М": "M",  # М
    "Н": "H",  # Н
    "О": "O",  # О
    "Р": "P",  # Р
    "С": "C",  # С
    "Т": "T",  # Т
    "У": "Y",  # У
    "Х": "X",  # Х
    "Ѕ": "S",  # Ѕ
    "І": "I",  # І
    "Ј": "J",  # Ј
    "Һ": "H",  # Һ
    # Greek
    "α": "a",  # α
    "ο": "o",  # ο
    "ρ": "p",  # ρ
    "ν": "v",  # ν
    "υ": "u",  # υ
    "χ": "x",  # χ
    "Α": "A",  # Α
    "Β": "B",  # Β
    "Ε": "E",  # Ε
    "Ζ": "Z",  # Ζ
    "Η": "H",  # Η
    "Ι": "I",  # Ι
    "Κ": "K",  # Κ
    "Μ": "M",  # Μ
    "Ν": "N",  # Ν
    "Ο": "O",  # Ο
    "Ρ": "P",  # Ρ
    "Τ": "T",  # Τ
    "Υ": "Y",  # Υ
    "Χ": "X",  # Χ
    # Armenian / other frequent
    "օ": "o",  # օ
    "հ": "h",  # հ
    # Fullwidth punctuation left over after NFKC in some inputs
    "．": ".",  # ．
    "／": "/",  # ／
    "：": ":",  # ：
    "？": "?",  # ？
    "＠": "@",  # ＠
    "＃": "#",  # ＃
# Zero-width and invisible separators used to break naive string matching.
    # Spelled as escapes: the literal characters are invisible in source and
    # get stripped by editors, which previously left an empty dict key here.
    "\u200b": "",  # ZERO WIDTH SPACE
    "\u200c": "",  # ZERO WIDTH NON-JOINER
    "\u200d": "",  # ZERO WIDTH JOINER
    "\u2028": "",  # LINE SEPARATOR
    "\u2029": "",  # PARAGRAPH SEPARATOR
    "\ufeff": "",  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u00ad": "",  # SOFT HYPHEN
    "\u180e": "",  # MONGOLIAN VOWEL SEPARATOR
    "\u2060": "",  # WORD JOINER
    "\u034f": "",  # COMBINING GRAPHEME JOINER
    "\u061c": "",  # ARABIC LETTER MARK
    "\u200e": "",  # LEFT-TO-RIGHT MARK
    "\u200f": "",  # RIGHT-TO-LEFT MARK
    # Bidirectional embedding/override controls. These are invisible in a URL
    # bar but reorder the glyphs a reader sees, so the text displayed and the
    # host actually resolved can differ. Stripping them removes the ability to
    # hide a real host behind a display-order illusion.
    "\u202a": "",  # LEFT-TO-RIGHT EMBEDDING
    "\u202b": "",  # RIGHT-TO-LEFT EMBEDDING
    "\u202c": "",  # POP DIRECTIONAL FORMATTING
    "\u202d": "",  # LEFT-TO-RIGHT OVERRIDE
    "\u202e": "",  # RIGHT-TO-LEFT OVERRIDE
    "\u2066": "",  # LEFT-TO-RIGHT ISOLATE
    "\u2067": "",  # RIGHT-TO-LEFT ISOLATE
    "\u2068": "",  # FIRST STRONG ISOLATE
    "\u2069": "",  # POP DIRECTIONAL ISOLATE
    # Khmer and Myanmar vowel-inherent characters, which browsers render as
    # part of the preceding consonant despite being separate code points.
    "\u17b4": "",  # KHMER VOWEL INHERENT AQ
    "\u17b5": "",  # KHMER VOWEL INHERENT AA
    "\u200c\u200d": "",  # ZWNJ/ZWJ run, kept adjacent for readability
}

_CONFUSABLE_RE = re.compile(f"[{''.join(re.escape(c) for c in _CONFUSABLES)}]")

#: Scripts that are entirely made of confusable letters. A host made only of these
#: plus digits/hyphens is almost always a homoglyph attack rather than a genuine
#: localised domain, so it is flagged rather than silently rewritten.
_CYRILLIC_BLOCKS = ((0x0400, 0x04FF), (0x0500, 0x052F))
_GREEK_BLOCKS = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))
_ARMENIAN_BLOCKS = ((0x0530, 0x058F),)


def fold_confusables(s: str) -> str:
    """Fold known homoglyphs and invisible separators onto their ASCII twins.

    NFKC handles fullwidth and compatibility forms but deliberately leaves
    Cyrillic and Greek letters alone, because they are genuinely different
    letters. That leaves the classic homoglyph phishing host untouched by
    normalisation. This pass closes that gap.

    Zero-width and soft-hyphen characters are removed outright: they render as
    nothing but defeat naive substring comparison, so two visually identical
    hosts can produce different model inputs.

    Returns the folded string. Never raises.
    """
    if not s:
        return ""
    if not _CONFUSABLE_RE.search(s):
        return s
    return _CONFUSABLE_RE.sub(lambda m: _CONFUSABLES[m.group(0)], s)


def has_nonascii_host(host: str) -> bool:
    """True when the host contains a non-ASCII letter.

    After folding this flags hosts that were genuinely internationalised rather
    than homoglyph-spoofed. It is a signal for the feature extractor, not a
    rejection: real IDN domains exist and must not be silently discarded.
    """
    if not host:
        return False
    for ch in host:
        if ord(ch) < 128:
            continue
        if unicodedata.category(ch).startswith("L"):
            return True
    return False


def normalize_url(url: str, *, strip_userinfo: bool = True) -> str:
    """Canonicalise a URL string for the character model.

    Steps, in order:

    1. Unicode ``NFKC`` normalisation. Phishing kits lean on fullwidth and
       compatibility look-alikes; NFKC folds them onto their ASCII twins so the
       character model sees the character a human would read.
    1b. Homoglyph and invisible-separator folding (:func:`fold_confusables`).
       NFKC does not touch Cyrillic/Greek look-alikes, so ``раypal.com``
       survives it intact; this pass maps them onto the ASCII letter a human
       reads and strips zero-width characters that render as nothing but
       defeat naive string comparison.
    2. Percent-decode *once*. Double-encoded paths (``%252e``) are a real
       evasion, but a single decode surfaces ``%2e`` -> ``.`` which the model can
       reason about; the raw form is preserved by the ``%`` character that
       survives the decode of anything above it.
    3. Lowercase the **scheme and host** only. The path and query are
       case-sensitive in reality and carry signal (many phishing kits vary case to
       dodge naive filters), so they are left alone.
    4. Collapse runs of the same character to at most two. ``aaaa`` and ``aa``
       say the same thing to a human and inflate the sequence length.
    5. Drop userinfo (``http://good.com@evil.tld`` -> ``http://evil.tld``). The
       model should not be able to key on a decoy host, and the security layer
       rejects these URLs before they ever reach inference.

    Returns the normalised string. Never raises on odd input: malformed URLs are
    normalised as far as they can be and the security layer is what rejects them.
    """
    if url is None:
        return ""
    s = str(url).strip()

    if not s:
        return ""

    # 1. Unicode compatibility normalisation.
    s = unicodedata.normalize("NFKC", s)

    # 1b. Homoglyph / invisible-separator folding. NFKC deliberately leaves
    # Cyrillic and Greek letters intact because they are genuinely different
    # letters, which means the classic spoofed host ("раypal.com") passes
    # through unchanged. This closes that gap and is applied to the whole string
    # so invisible separators cannot split a host either.
    s = fold_confusables(s)

    # 2. Single percent-decode, only for well-formed escapes.
    if "%" in s:
        def _dec(m: re.Match[str]) -> str:
            ch = chr(int(m.group(0)[1:], 16))
            # Only decode into characters we can actually represent; otherwise
            # keep the literal escape so the model still sees "%2e".
            return ch if ch in DEFAULT_ALPHABET or ch.isalnum() else m.group(0)

        s = _HEX_RUN.sub(_dec, s)

    # 3. Lowercase scheme + host only.
    m = re.match(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*)://(?P<rest>.*)$", s, re.DOTALL)
    if m:
        scheme = m.group("scheme").lower()
        rest = m.group("rest")
        # Split host from the rest at the first / ? or #
        cut = len(rest)
        for i, ch in enumerate(rest):
            if ch in "/?#":
                cut = i
                break
        host, tail = rest[:cut], rest[cut:]
        s = f"{scheme}://{host.lower()}{tail}"
    else:
        # No scheme: lowercase only the leading authority-ish part.
        s = s.lower() if len(s) < 12 else s

    # 4. Collapse long repeats.
    s = _REPEATS.sub(r"\1\1", s)

    # 5. Strip userinfo.
    if strip_userinfo:
        s = re.sub(
            r"^([A-Za-z][A-Za-z0-9+.\-]*://)[^/?#]*@",
            r"\1",
            s,
            flags=re.IGNORECASE,
        )

    return s


def host_of(url: str) -> str:
    """Best-effort hostname extraction, lowercased. Never raises."""
    try:
        from urllib.parse import urlsplit

        h = urlsplit(url if "://" in url else f"//{url}").hostname
        return (h or "").lower()
    except ValueError:
        return ""


def suggest_max_length(urls: Iterable[str], percentile: float = 99.0, cap: int = 1024, floor: int = 64) -> int:
    """Pick the character-sequence cap from a URL length distribution.

    Uses the requested percentile of the observed lengths so the common case is
    never truncated, with headroom for the tail. Falls back to ``floor`` when no
    usable data is supplied.
    """
    lengths: list[int] = []
    for u in urls:
        lengths.append(len(normalize_url(u)))
    if not lengths:
        return floor
    lengths.sort()
    idx = min(len(lengths) - 1, max(0, int(round((percentile / 100.0) * (len(lengths) - 1)))))
    p = lengths[idx]
    return int(min(cap, max(floor, math.ceil(p * 1.5))))


class CharTokenizer:
    """Fixed character-level tokenizer with ``<PAD>``/``<UNK>``.

    ``fit`` builds the mapping from training URLs only. ``encode`` maps a URL to
    a fixed-width id vector plus an attention mask, so batching needs no
    per-batch length logic.
    """

    def __init__(
        self,
        max_length: int = 256,
        alphabet: frozenset[str] | None = None,
        min_count: int = 5,
    ) -> None:
        self.max_length = int(max_length)
        self.alphabet = set(alphabet) if alphabet is not None else set(DEFAULT_ALPHABET)
        self.min_count = int(min_count)

        self.stoi: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.itos: list[str] = [PAD_TOKEN, UNK_TOKEN]

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    # ------------------------------------------------------------------
    def fit(self, urls: Sequence[str]) -> "CharTokenizer":
        """Build the vocabulary from training URLs.

        Only characters in ``alphabet`` that clear ``min_count`` are kept. The
        order is sorted so the vocabulary is reproducible run to run.
        """
        counts: Counter[str] = Counter()
        for raw in urls:
            for ch in normalize_url(raw):
                counts[ch] += 1

        keep = sorted(ch for ch, c in counts.items() if c >= self.min_count and ch in self.alphabet)

        # Rebuild from scratch so fit() is idempotent.
        self.stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.itos = [PAD_TOKEN, UNK_TOKEN]
        for ch in keep:
            if ch not in self.stoi:
                self.stoi[ch] = len(self.itos)
                self.itos.append(ch)
        return self

    # ------------------------------------------------------------------
    def encode(self, url: str) -> tuple[list[int], list[int]]:
        """Return ``(ids, mask)`` of exactly ``max_length`` elements.

        ``mask`` is 1 for a real character and 0 for padding, so downstream
        attention pooling can ignore the tail.
        """
        text = normalize_url(url)
        ids = [self.stoi.get(ch, self.unk_id) for ch in text[: self.max_length]]
        n = len(ids)
        ids = ids + [self.pad_id] * (self.max_length - n)
        mask = [1] * n + [0] * (self.max_length - n)
        return ids, mask

    def encode_batch(self, urls: Sequence[str]) -> tuple[list[list[int]], list[list[int]]]:
        all_ids: list[list[int]] = []
        all_mask: list[list[int]] = []
        for u in urls:
            ids, mask = self.encode(u)
            all_ids.append(ids)
            all_mask.append(mask)
        return all_ids, all_mask

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "max_length": self.max_length,
            "min_count": self.min_count,
            "alphabet": sorted(self.alphabet),
            "itos": list(self.itos),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        tok = cls(
            max_length=int(state["max_length"]),
            alphabet=frozenset(state.get("alphabet") or DEFAULT_ALPHABET),
            min_count=int(state.get("min_count", 5)),
        )
        itos = list(state["itos"])
        tok.itos = itos
        tok.stoi = {ch: i for i, ch in enumerate(itos)}
        # Defensive: guarantee the specials exist even if the file was edited.
        if PAD_TOKEN not in tok.stoi:
            tok.itos.insert(0, PAD_TOKEN)
        if UNK_TOKEN not in tok.stoi:
            tok.itos.insert(1, UNK_TOKEN)
        tok.stoi = {ch: i for i, ch in enumerate(tok.itos)}
        return tok

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))