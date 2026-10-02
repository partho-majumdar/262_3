"""Domain/IP relationship graph for the phishing GNN.

Motivation
----------
The URL, HTML and vision branches in this project all look at *one URL in
isolation*. A real phishing campaign, however, is a **fleet**: many registrable
domains registered together, pointed at the same hosting provider, serving the
same login-kit path prefix. A domain that shares that infrastructure with a
known-phishing domain is itself suspicious even when its own string looks bland.
This module turns a list of URLs into exactly that relationship graph so a GNN
can propagate that evidence (``training/train_gnn.py``).

The no-label-leakage rule (read this before adding an edge type)
----------------------------------------------------------------
**No label may ever influence edge construction.** Concretely, this module:

* receives ``labels`` only to attach a *target* to the node a URL already
  created, and
* builds **every** edge from URL/hostname string structure only - shared IPv4
  literal, shared /24 subnet, shared subdomain path token, shared TLD, or
  token containment.

It deliberately does **not** create an edge between "two domains that happen to
share a label". Such an edge would hand the model the answer: message passing
over a label-homogeneous neighbourhood collapses to reading the neighbourhood's
label, and the reported accuracy would be an artefact of the graph builder, not
of phishing structure. The test
``test_edge_set_identical_under_label_permutation`` in
``tests/test_graph_model.py`` enforces this directly: permuting the label vector
must leave the edge set byte-identical.

The same principle rules out using PhiUSIIL's page-derived columns or the
split label distribution for anything structural.

Honest limitation (prominent because it dominates the results)
------------------------------------------------------------
**This graph is built from URL strings alone. There is no live DNS resolution.**
Consequently the overwhelming majority of domains have **no** IP edge at all:
only URLs whose host is already an IPv4 literal produce an IP node. The graph is
therefore *mostly domain<->domain structural edges* (shared subdomain token,
shared TLD, token containment), and the IP/subnet relations are exercised only
when a caller supplies ``extra_domains`` resolution data (see
``build_graph(..., extra_domains=...)``). Adding a resolver (or a historical
passive-DNS table) is the single highest-value change to this component, and it
is the reason the reported metrics should be read as a floor, not a ceiling.

Determinism
-----------
Every feature and every index assignment is a pure function of the input
strings. ``node_features`` uses :func:`zlib.crc32` for its one hash-derived
feature - never Python's ``hash()``, which is salted per process and would make
a saved graph unfalsifiable.
"""

from __future__ import annotations

import json
import math
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

from app.preprocessing.url_features import BRAND_TOKENS

__all__ = [
    "GRAPH_FEATURE_NAMES",
    "KIND_DOMAIN",
    "KIND_IP",
    "KIND_SUBNET",
    "EDGE_DOMAIN_IP",
    "EDGE_DOMAIN_SUBNET",
    "EDGE_SHARED_SUBDOMAIN",
    "EDGE_SHARED_TLD",
    "EDGE_CONTAINMENT",
    "EDGE_SHARED_IP",
    "EDGE_IP_SUBNET",
    "EDGE_TYPE_NAMES",
    "N_EDGE_TYPES",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SPLIT_TEST",
    "SPLIT_UNASSIGNED",
    "GraphSpec",
    "build_graph",
    "node_features",
    "save_graph",
    "load_graph",
    "default_graph_path",
    "host_of",
    "registered_domain_of",
    "node_feature_matrix",
    "subnet_of",
]


# ---------------------------------------------------------------------------
# Vocabulary: node kinds, edge types, split codes
# ---------------------------------------------------------------------------
KIND_DOMAIN = 0
KIND_IP = 1
KIND_SUBNET = 2

KIND_NAMES: dict[int, str] = {KIND_DOMAIN: "domain", KIND_IP: "ip", KIND_SUBNET: "subnet"}

#: domain -> IP, from a URL whose host is an IPv4 literal or from supplied
#: ``extra_domains`` resolution data.
EDGE_DOMAIN_IP = 0
#: domain -> /24 subnet node, same evidence, one level of aggregation coarser.
EDGE_DOMAIN_SUBNET = 1
#: domain <-> domain: the two URLs used the same subdomain path token
#: (``login.evil-a.tld`` and ``login.evil-b.tld``).
EDGE_SHARED_SUBDOMAIN = 2
#: domain <-> domain: same registered TLD (degree-capped; see ``max_tld_degree``).
EDGE_SHARED_TLD = 3
#: domain <-> domain: a token of one domain's registrable label also appears as a
#: subdomain token of the other (``secure.paypal-verify.tld`` vs ``secure.b.tld``).
EDGE_CONTAINMENT = 4
#: domain <-> domain: both resolve to the same IP. Redundant with the 2-hop path
#: through the IP node, but kept explicit because it is the relation the project
#: actually cares about ("shared infrastructure") and because it survives a
#: downstream decision to drop the IP nodes.
EDGE_SHARED_IP = 5
#: ip -> subnet. Emitted for every IP node, including IPs that came from an
#: IP-literal URL, so subnet nodes are never isolated.
EDGE_IP_SUBNET = 6

EDGE_TYPE_NAMES: dict[int, str] = {
    EDGE_DOMAIN_IP: "domain_ip",
    EDGE_DOMAIN_SUBNET: "domain_subnet",
    EDGE_SHARED_SUBDOMAIN: "shared_subdomain",
    EDGE_SHARED_TLD: "shared_tld",
    EDGE_CONTAINMENT: "containment",
    EDGE_SHARED_IP: "shared_ip",
    EDGE_IP_SUBNET: "ip_subnet",
}
N_EDGE_TYPES = len(EDGE_TYPE_NAMES)

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2
SPLIT_UNASSIGNED = -1

SPLIT_CODES: dict[str, int] = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST}

#: IPv4 space is exhausted by /24s in the sense that mass phishing hosting is
#: contiguous; /24 keeps the subnet nodes meaningful without exploding the graph.
SUBNET_PREFIX = 24


# ---------------------------------------------------------------------------
# Host parsing
# ---------------------------------------------------------------------------
def _tld_extractor() -> Any:
    """Cached offline ``tldextract`` extractor.

    ``suffix_list_urls=()`` disables the network fetch so graph construction is
    reproducible offline and inside tests; the same setting is used by
    ``training/make_splits.py`` so both stages agree on the registrable domain.
    """
    extractor = getattr(_tld_extractor, "_extractor", None)
    if extractor is None:
        import tldextract

        extractor = tldextract.TLDExtract(suffix_list_urls=())
        _tld_extractor._extractor = extractor  # type: ignore[attr-defined]
    return extractor


def registered_domain_of(host: str) -> str:
    """Registrable domain for ``host`` (lowercased), ``""`` when unparseable."""
    if not host:
        return ""
    reg = _tld_extractor()(host).registered_domain
    return reg.lower() if reg else host.strip().lower()


def _suffix_and_subdomain(host: str) -> tuple[str, str]:
    """``(suffix, subdomain)`` for ``host`` using the bundled PSL snapshot."""
    ext = _tld_extractor()(host)
    return (ext.suffix.lower(), ext.subdomain.lower())


def host_of(url: str) -> str:
    """Lowercased hostname of ``url``, ``""`` when unparseable.

    ``urlsplit`` raises ``ValueError`` on some malformed IPv6-ish hosts, and a
    single bad row must not abort a 200k-row build.
    """
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    return host.strip().lower().rstrip(".")


def _is_ipv4(token: str) -> bool:
    try:
        ip_address(token)
    except ValueError:
        return False
    return True


def subnet_of(ip: str) -> str:
    """The /24 subnet string for an IPv4 literal."""
    return str(ip_network(f"{ip}/{SUBNET_PREFIX}", strict=False))


def _tokenise(label: str) -> list[str]:
    """Split a hostname label into lowercased alphanumeric tokens.

    ``paypal-secure_01`` -> ``["paypal", "secure", "01"]``. Tokenising is what
    makes the containment relation meaningful: a phisher who cannot use the real
    brand still puts its token somewhere in the name.
    """
    out: list[str] = []
    buf: list[str] = []
    for ch in label:
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return [t.lower() for t in out if t]


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------
#: Ordered node-feature names. Saved into the ``.npz`` so inference cannot
#: silently disagree with the graph a checkpoint was trained on.
GRAPH_FEATURE_NAMES: tuple[str, ...] = (
    "label_count",
    "char_length",
    "digit_ratio",
    "hyphen_count",
    "underscore_count",
    "max_digit_run",
    "char_entropy",
    "is_ipv4",
    "is_punycode",
    "tld_length",
    "has_brand_token",
    "brand_token_count",
    "max_label_length",
    "mean_label_length",
    "alphabetic_ratio",
    "crc32_unit",
    "is_subnet",
    "subnet_prefix_len",
)

N_FEATURES = len(GRAPH_FEATURE_NAMES)

_BRAND_SET = tuple(b.lower() for b in BRAND_TOKENS)


def node_features(domain_or_ip: str) -> list[float]:
    """Deterministic handcrafted features for one node name.

    Every value is a pure function of the string: no randomness, no network, no
    dataset statistics. The single hash-derived value (``crc32_unit``) uses
    :func:`zlib.crc32`, whose result is stable across processes, platforms and
    Python versions - unlike the builtin ``hash()``, which is salted per process
    and would make a saved graph irreproducible.

    The features are deliberately *string-shape* features (length, digit ratio,
    token entropy, punycode, brand-token presence) because that is the part of
    the signal available without DNS. They mirror the URL branch's
    handcrafted block, but are computed on the registrable domain / IP alone.
    """
    raw = (domain_or_ip or "").strip().lower()
    is_subnet = 1.0 if "/" in raw else 0.0
    prefix_len = 0.0
    if is_subnet:
        addr, _, plen = raw.partition("/")
        try:
            prefix_len = float(plen)
        except ValueError:
            prefix_len = 0.0
        raw = addr
    if not raw:
        raw = ""

    labels = [lab for lab in raw.split(".") if lab]
    n_chars = len(raw)
    char_counts = Counter(raw)
    entropy = 0.0
    if n_chars > 0:
        for c in char_counts.values():
            p = c / n_chars
            entropy -= p * math.log2(p)
    entropy_norm = entropy / math.log2(2) if n_chars else 0.0  # bits -> 0..1-ish

    max_digit_run = 0
    run = 0
    for ch in raw:
        if ch.isdigit():
            run += 1
            max_digit_run = max(max_digit_run, run)
        else:
            run = 0

    if is_subnet or _is_ipv4(raw):
        tld_len = 0.0
    else:
        suffix, _ = _suffix_and_subdomain(raw)
        tld_len = float(len(suffix))

    brand_count = sum(1 for b in _BRAND_SET if b in raw)
    label_lens = [len(lab) for lab in labels] or [0]
    alpha = sum(1 for ch in raw if ch.isalpha())

    return [
        float(len(labels)),
        float(n_chars),
        (sum(1 for ch in raw if ch.isdigit()) / n_chars) if n_chars else 0.0,
        float(raw.count("-")),
        float(raw.count("_")),
        float(max_digit_run),
        float(entropy_norm),
        1.0 if (not is_subnet and _is_ipv4(raw)) else 0.0,
        1.0 if "xn--" in raw else 0.0,
        tld_len,
        1.0 if brand_count > 0 else 0.0,
        float(brand_count),
        float(max(label_lens)),
        float(sum(label_lens) / len(label_lens)),
        (alpha / n_chars) if n_chars else 0.0,
        zlib.crc32(raw.encode("utf-8")) / 4294967296.0,
        is_subnet,
        prefix_len,
    ]


def node_feature_matrix(names: Sequence[str]) -> np.ndarray:
    """``(len(names), N_FEATURES)`` float32 matrix, row-aligned with ``names``."""
    out = np.zeros((len(names), N_FEATURES), dtype=np.float32)
    for i, name in enumerate(names):
        out[i] = np.asarray(node_features(name), dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# The graph container
# ---------------------------------------------------------------------------
@dataclass
class GraphSpec:
    """An immutable-by-convention description of the built graph.

    Node indices are laid out in three contiguous blocks so a caller can slice a
    block without a lookup table: domains first (``0 .. n_domain-1``), then IP
    literals, then /24 subnet nodes.

    Attributes:
        domain_to_index: registrable domain -> node index.
        ip_to_index: IPv4 literal -> node index.
        subnet_to_index: ``a.b.c.d/24`` -> node index.
        node_names: node index -> canonical name.
        node_kind: ``(N,)`` int64, one of ``KIND_*``.
        edge_index: ``(2, E)`` int64. Row 0 is the **source**, row 1 the
            **target**; message passing flows ``source -> target``. Undirected
            relations are emitted in both directions.
        edge_type: ``(E,)`` int64, one of the ``EDGE_*`` codes.
        x: ``(N, N_FEATURES)`` float32, label-free node features.
        y: ``(N,)`` int64 node labels, ``-1`` where no URL labels the node.
        node_split: ``(N,)`` int64, ``SPLIT_TRAIN/VAL/TEST`` or
            ``SPLIT_UNASSIGNED``. Assigned only from the split of the URL that
            created the node - never recomputed here.
        label_conflicts: how many URLs disagreed about a node's label. A grouped
            split should make this 0; it is reported rather than assumed.
    """

    domain_to_index: dict[str, int]
    ip_to_index: dict[str, int]
    subnet_to_index: dict[str, int]
    node_names: list[str]
    node_kind: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    x: np.ndarray
    y: np.ndarray
    node_split: np.ndarray
    label_conflicts: int = 0

    # -- sizes ----------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_domain_nodes(self) -> int:
        return len(self.domain_to_index)

    @property
    def num_ip_nodes(self) -> int:
        return len(self.ip_to_index)

    @property
    def num_subnet_nodes(self) -> int:
        return len(self.subnet_to_index)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[1])

    # -- masks ---------------------------------------------------------
    def labelled_mask(self) -> np.ndarray:
        """``(N,)`` bool - nodes that at least one URL labels."""
        return self.y >= 0

    def split_mask(self, split_code: int) -> np.ndarray:
        """``(N,)`` bool - nodes belonging to ``split_code``."""
        return self.node_split == split_code

    def labelled_split_mask(self, split_code: int) -> np.ndarray:
        """``(N,)`` bool - labelled nodes of ``split_code``."""
        return (self.node_split == split_code) & (self.y >= 0)

    def edge_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {name: 0 for name in EDGE_TYPE_NAMES.values()}
        for t in np.unique(self.edge_type):
            counts[EDGE_TYPE_NAMES[int(t)]] = int((self.edge_type == t).sum())
        return counts

    def edge_set(self) -> set[tuple[int, int]]:
        """Directed edge pairs as a python set - used by the leakage tests."""
        src = self.edge_index[0]
        dst = self.edge_index[1]
        return {(int(a), int(b)) for a, b in zip(src, dst)}

    def named_edge_set(self) -> set[tuple[str, str, int]]:
        """``(source_name, target_name, edge_type)`` triples.

        Index-level comparison is only meaningful against another build with the
        same node ordering; comparing names makes the no-leakage test robust to
        that.
        """
        return {
            (self.node_names[int(a)], self.node_names[int(b)], int(t))
            for a, b, t in zip(self.edge_index[0], self.edge_index[1], self.edge_type)
        }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def _bucket(candidates: Sequence[str], limit: int | None) -> list[str]:
    """Deterministically trim a candidate list to ``limit`` entries.

    Sorting by name (not by dict or set order) makes the retained neighbourhood a
    pure function of the node set, so two builds of the same data produce the
    same edges - a prerequisite for the reproducibility claim.
    """
    ordered = sorted(candidates)
    if limit is None or len(ordered) <= limit:
        return ordered
    return ordered[:limit]


def build_graph(
    urls: Sequence[str],
    labels: Sequence[int],
    *,
    extra_domains: Mapping[str, Sequence[str]] | None = None,
    splits: Sequence[str] | None = None,
    include_subnet_nodes: bool = True,
    include_shared_tld: bool = True,
    include_shared_subdomain: bool = True,
    include_containment: bool = True,
    max_tld_degree: int | None = 8,
    max_shared_degree: int | None = 32,
) -> GraphSpec:
    """Build the domain/IP relationship graph from URLs.

    Args:
        urls: raw URL strings, any order.
        labels: 0/1 per URL, phishing = 1. Used **only** as node targets.
        extra_domains: optional ``{host_or_domain: [ip, ...]}`` resolution data.
            This is the only way a non-IP-literal domain acquires an IP or subnet
            edge, and the shipped dataset has none - see the module docstring.
        splits: optional ``"train"/"val"/"test"`` per URL, copied onto the node
            the URL creates. Since the project's split is grouped by registered
            domain, every URL behind a node shares one split, so this reproduces
            the domain grouping exactly rather than re-deriving it.
        include_subnet_nodes: add /24 subnet nodes.
        include_shared_tld: add the (degree-capped) same-TLD relation.
        include_shared_subdomain: add the shared-subdomain-token relation.
        include_containment: add the label-token/subdomain-token relation.
        max_tld_degree: cap on same-TLD neighbours per domain. Without a cap a
            popular TLD such as ``.com`` produces a quadratic edge count.
        max_shared_degree: cap on the other domain<->domain relations.

    Returns:
        A :class:`GraphSpec`. Node ordering is deterministic: domains sorted
        alphabetically, then IPs, then subnets, each in sorted order.
    """
    if len(urls) != len(labels):
        raise ValueError(f"urls ({len(urls)}) and labels ({len(labels)}) must be the same length")
    if splits is not None and len(splits) != len(urls):
        raise ValueError(f"splits ({len(splits)}) must be the same length as urls ({len(urls)})")

    resolution: dict[str, set[str]] = defaultdict(set)
    for key, ips in (extra_domains or {}).items():
        k = str(key).strip().lower()
        for ip in ips:
            ip = str(ip).strip()
            if ip and _is_ipv4(ip):
                resolution[k].add(ip)

    # --- pass 1: which node does each URL create? -----------------------
    node_kind_of_url: list[tuple[str, int]] = []
    domain_subdomain_tokens: dict[str, set[str]] = defaultdict(set)
    domain_label_tokens: dict[str, set[str]] = defaultdict(set)
    domain_suffix: dict[str, str] = {}
    domain_ips: dict[str, set[str]] = defaultdict(set)
    literal_ips: set[str] = set()

    for url in urls:
        host = host_of(url)
        if not host:
            node_kind_of_url.append(("", -1))
            continue
        if _is_ipv4(host):
            # An IP-literal URL has no registrable domain, so its **IP node is
            # the labelled node**. The IP is still linked to every domain that
            # resolves to it, which is the domain->IP edge the design wants; the
            # label simply lands on the IP side rather than on a synthetic
            # domain node that the string does not justify.
            literal_ips.add(host)
            node_kind_of_url.append((host, KIND_IP))
            continue
        reg = registered_domain_of(host)
        if not reg:
            node_kind_of_url.append(("", -1))
            continue
        suffix, subdomain = _suffix_and_subdomain(host)
        domain_suffix[reg] = suffix
        if subdomain:
            for tok in _tokenise(subdomain.replace(".", " ")):
                domain_subdomain_tokens[reg].add(tok)
        label_tokens = set(_tokenise(reg.split(".", 1)[0]))
        domain_label_tokens[reg].update(label_tokens)
        node_kind_of_url.append((reg, KIND_DOMAIN))

    # Resolution data keyed by host or by registrable domain.
    for key, ips in resolution.items():
        for host_key in (key, registered_domain_of(key)):
            if host_key in domain_suffix:
                domain_ips[host_key].update(ips)

    # --- pass 2: deterministic node indexing ---------------------------
    domains = sorted(domain_suffix)
    domain_to_index = {d: i for i, d in enumerate(domains)}
    all_ips: set[str] = {ip for ip in literal_ips if _is_ipv4(ip)}
    all_ips.update(key for key in resolution if _is_ipv4(key))
    all_ips.update(ip for ips in resolution.values() for ip in ips if _is_ipv4(ip))
    all_ips.update(ip for d in domains for ip in domain_ips.get(d, ()) if _is_ipv4(ip))
    ips_sorted = sorted(all_ips)
    ip_to_index = {ip: len(domains) + i for i, ip in enumerate(ips_sorted)}

    subnets_sorted: list[str] = []
    subnet_to_index: dict[str, int] = {}
    if include_subnet_nodes:
        wanted = sorted({subnet_of(ip) for ip in ips_sorted})
        offset = len(domains) + len(ips_sorted)
        subnet_to_index = {s: offset + i for i, s in enumerate(wanted)}
        subnets_sorted = wanted

    node_names = list(domains) + list(ips_sorted) + list(subnets_sorted)
    node_kind = np.array(
        [KIND_DOMAIN] * len(domains) + [KIND_IP] * len(ips_sorted) + [KIND_SUBNET] * len(subnets_sorted),
        dtype=np.int64,
    )

    # --- pass 3: labels and splits -------------------------------------
    y = np.full(len(node_names), -1, dtype=np.int64)
    node_split = np.full(len(node_names), SPLIT_UNASSIGNED, dtype=np.int64)
    all_maps: dict[int, dict[str, int]] = {
        KIND_DOMAIN: domain_to_index,
        KIND_IP: ip_to_index,
        KIND_SUBNET: subnet_to_index,
    }
    conflicts = 0
    for i, (key, kind) in enumerate(node_kind_of_url):
        if not key:
            continue
        idx = all_maps[kind][key]
        lab = int(labels[i])
        if y[idx] == -1:
            y[idx] = lab
        elif y[idx] != lab:
            conflicts += 1
        if splits is not None:
            code = SPLIT_CODES.get(str(splits[i]).strip().lower(), SPLIT_UNASSIGNED)
            if node_split[idx] == SPLIT_UNASSIGNED:
                node_split[idx] = code

    # --- pass 4: edges --------------------------------------------------
    src: list[int] = []
    dst: list[int] = []
    etype: list[int] = []
    seen: set[tuple[int, int, int]] = set()

    def link(a: int, b: int, t: int) -> None:
        if a == b:
            return
        key = (a, b, t) if a < b else (b, a, t)
        if key in seen:
            return
        seen.add(key)
        src.extend([a, b])
        dst.extend([b, a])
        etype.extend([t, t])

    # 4a. domain -> ip / subnet, plus the explicit shared-ip domain<->domain link
    ip_owners: dict[str, list[str]] = defaultdict(list)
    for domain in domains:
        for ip in sorted(domain_ips.get(domain, ())):
            if ip not in ip_to_index:
                continue
            link(domain_to_index[domain], ip_to_index[ip], EDGE_DOMAIN_IP)
            ip_owners[ip].append(domain)
            if include_subnet_nodes:
                sub = subnet_of(ip)
                if sub in subnet_to_index:
                    link(domain_to_index[domain], subnet_to_index[sub], EDGE_DOMAIN_SUBNET)
    for ip, owners in ip_owners.items():
        capped = _bucket(owners, max_shared_degree)
        for i, a in enumerate(capped):
            for b in capped[i + 1 :]:
                link(domain_to_index[a], domain_to_index[b], EDGE_SHARED_IP)

    # 4a-bis. ip -> subnet for every IP node, so a subnet node is never an
    # isolated island. This is the only edge a bare IP-literal URL contributes
    # beyond its own node.
    if include_subnet_nodes:
        for ip in ips_sorted:
            sub = subnet_of(ip)
            if sub in subnet_to_index:
                link(ip_to_index[ip], subnet_to_index[sub], EDGE_IP_SUBNET)

    # 4b. shared subdomain token
    if include_shared_subdomain:
        token_index: dict[str, list[str]] = defaultdict(list)
        for domain, tokens in domain_subdomain_tokens.items():
            for tok in sorted(tokens):
                token_index[tok].append(domain)
        for tok in sorted(token_index):
            members = _bucket(token_index[tok], max_shared_degree)
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    link(domain_to_index[a], domain_to_index[b], EDGE_SHARED_SUBDOMAIN)

    # 4c. token containment: a token of A's registrable label sits in B's path
    if include_containment:
        sub_index: dict[str, list[str]] = defaultdict(list)
        for domain, tokens in domain_subdomain_tokens.items():
            for tok in sorted(tokens):
                sub_index[tok].append(domain)
        for domain in domains:
            for tok in sorted(domain_label_tokens.get(domain, ())):
                for other in _bucket(sub_index.get(tok, []), max_shared_degree):
                    link(domain_to_index[domain], domain_to_index[other], EDGE_CONTAINMENT)

    # 4d. shared TLD (degree-capped, deterministic)
    if include_shared_tld:
        suffix_index: dict[str, list[str]] = defaultdict(list)
        for domain in domains:
            suffix_index[domain_suffix.get(domain, "")].append(domain)
        for suf in sorted(suffix_index):
            members = _bucket(suffix_index[suf], max_tld_degree)
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    link(domain_to_index[a], domain_to_index[b], EDGE_SHARED_TLD)

    edge_index = (
        np.array([src, dst], dtype=np.int64)
        if src
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_type = np.array(etype, dtype=np.int64)

    order = np.lexsort((edge_type, edge_index[1], edge_index[0]))
    edge_index = edge_index[:, order]
    edge_type = edge_type[order]

    spec = GraphSpec(
        domain_to_index=domain_to_index,
        ip_to_index=ip_to_index,
        subnet_to_index=subnet_to_index,
        node_names=node_names,
        node_kind=node_kind,
        edge_index=edge_index,
        edge_type=edge_type,
        x=node_feature_matrix(node_names),
        y=y,
        node_split=node_split,
        label_conflicts=conflicts,
    )
    return spec


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_graph(spec: GraphSpec, path: Path, *, extra_meta: Mapping[str, Any] | None = None) -> Path:
    """Write ``spec`` to a compressed ``.npz`` under ``data/graph/``.

    The dict-valued index maps go in as one JSON blob (numpy arrays cannot hold
    string-keyed dicts) alongside a schema version, so a later loader can refuse
    a file written by an incompatible version instead of misreading it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "schema_version": 1,
        "domain_to_index": spec.domain_to_index,
        "ip_to_index": spec.ip_to_index,
        "subnet_to_index": spec.subnet_to_index,
        "node_names": spec.node_names,
        "feature_names": list(GRAPH_FEATURE_NAMES),
        "label_conflicts": int(spec.label_conflicts),
        "extra": dict(extra_meta or {}),
    }
    np.savez_compressed(
        path,
        edge_index=spec.edge_index,
        edge_type=spec.edge_type,
        node_kind=spec.node_kind,
        x=spec.x,
        y=spec.y,
        node_split=spec.node_split,
        meta_json=np.array(json.dumps(meta), dtype=object),
    )
    return path


def load_graph(path: Path) -> GraphSpec:
    """Read a graph written by :func:`save_graph`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing graph file {path}; run training/train_gnn.py first")
    with np.load(path, allow_pickle=True) as blob:
        meta = json.loads(str(blob["meta_json"].item()))
        if int(meta.get("schema_version", 0)) != 1:
            raise ValueError(
                f"{path} was written with graph schema {meta.get('schema_version')}; this "
                "loader understands version 1 only. Rebuild the graph."
            )
        spec = GraphSpec(
            domain_to_index={str(k): int(v) for k, v in meta["domain_to_index"].items()},
            ip_to_index={str(k): int(v) for k, v in meta["ip_to_index"].items()},
            subnet_to_index={str(k): int(v) for k, v in meta["subnet_to_index"].items()},
            node_names=[str(n) for n in meta["node_names"]],
            node_kind=blob["node_kind"].astype(np.int64),
            edge_index=blob["edge_index"].astype(np.int64),
            edge_type=blob["edge_type"].astype(np.int64),
            x=blob["x"].astype(np.float32),
            y=blob["y"].astype(np.int64),
            node_split=blob["node_split"].astype(np.int64),
            label_conflicts=int(meta.get("label_conflicts", 0)),
        )
    if list(meta.get("feature_names", [])) != list(GRAPH_FEATURE_NAMES):
        raise ValueError(
            f"{path} was built with a different feature vector; rebuild the graph rather "
            "than feeding a mismatched feature matrix to the model"
        )
    return spec


def default_graph_path(root: Path | None = None) -> Path:
    """``<backend>/data/graph/domain_ip_graph.npz``."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return root / "data" / "graph" / "domain_ip_graph.npz"
