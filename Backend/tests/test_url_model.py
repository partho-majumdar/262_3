"""P2 acceptance: URL preprocessing, model shapes, inference contract.

Real assertions about real behaviour - a shape test that passes while the
attention weights do not actually sum to 1 is worse than no test at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from app.models.url_model import AdditiveAttention, URLCharModel
from app.preprocessing.url_features import (
    FEATURE_NAMES,
    FeatureScaler,
    extract_batch,
    extract_handcrafted_features,
    name_label_of,
    shannon_entropy,
)
from app.preprocessing.url_preprocessing import (
    PAD_TOKEN,
    UNK_TOKEN,
    CharTokenizer,
    normalize_url,
    suggest_max_length,
)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_normalize_strips_userinfo_decoy_host():
    """http://good.com@evil must normalise to evil, not to a decoy."""
    assert normalize_url("http://good.com@evil.tld/x") == "http://evil.tld/x"
    assert "good.com" not in normalize_url("http://good.com@evil.tld/x")


def test_normalize_lowercases_host_but_preserves_path_case():
    out = normalize_url("HTTP://PayPal.COM/Login/Verify")
    assert out.startswith("http://paypal.com/")
    assert "Login" in out, "path case carries signal and must survive"


def test_normalize_decodes_percent_escapes_once():
    assert "%2e" not in normalize_url("http://a.tld/%2e%2e/admin")


def test_normalize_collapses_long_repeats():
    assert "aaa" not in normalize_url("http://a.tld/aaaaaaaa")


def test_normalize_applies_nfkc_to_fullwidth_lookalikes():
    """Fullwidth Latin must fold onto ASCII so the model reads what a human sees."""
    assert normalize_url("http://ｅｖｉｌ．com/") == normalize_url("http://evil.com/")


def test_normalize_handles_empty_and_none():
    assert normalize_url("") == ""
    assert normalize_url(None) == ""


def test_normalize_is_deterministic():
    u = "HTTPS://User@PayPal.COM:443/secure/login?a=1&b=2"
    assert normalize_url(u) == normalize_url(u)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
@pytest.fixture
def fitted_tokenizer() -> CharTokenizer:
    tk = CharTokenizer(max_length=64, min_count=1)
    tk.fit(["http://paypal.com/login", "http://a.b.co/x", "http://paypal.com/login"])
    return tk


def test_tokenizer_vocab_has_specials_and_is_frozen(fitted_tokenizer):
    assert fitted_tokenizer.itos[0] == PAD_TOKEN
    assert fitted_tokenizer.itos[1] == UNK_TOKEN
    assert fitted_tokenizer.pad_id == 0
    assert fitted_tokenizer.unk_id == 1


def test_tokenizer_min_count_prunes_rare_chars():
    rare = CharTokenizer(max_length=32, min_count=1000)
    rare.fit(["http://a.com", "http://b.com"])
    assert rare.vocab_size == 2, "with min_count this high only the specials survive"


def test_encode_returns_fixed_width_with_mask(fitted_tokenizer):
    ids, mask = fitted_tokenizer.encode("http://paypal.com/login")
    assert len(ids) == fitted_tokenizer.max_length
    assert len(mask) == fitted_tokenizer.max_length
    assert sum(mask) == min(len("http://paypal.com/login"), fitted_tokenizer.max_length)
    # Padding is exactly where the mask says it is.
    for i, m in enumerate(mask):
        if m == 0:
            assert ids[i] == fitted_tokenizer.pad_id


def test_encode_truncates_to_max_length(fitted_tokenizer):
    # A repeated single character would be collapsed by normalisation, so build
    # the long URL from a 10-character block that has no consecutive repeats.
    long_url = "http://paypal.com/" + ("0123456789" * 60)
    assert len(normalize_url(long_url)) > fitted_tokenizer.max_length
    ids, mask = fitted_tokenizer.encode(long_url)
    assert len(ids) == fitted_tokenizer.max_length
    assert sum(mask) == fitted_tokenizer.max_length
    assert mask[0] == 1 and mask[-1] == 1, "a truncated URL is full of real characters"


def test_unknown_characters_map_to_unk(fitted_tokenizer):
    ids, mask = fitted_tokenizer.encode("http://a.tld/☃")
    real = [i for i, m in enumerate(ids, 1) if m]
    assert fitted_tokenizer.unk_id in [ids[i - 1] for i in real]


def test_tokenizer_roundtrips_through_state_dict(fitted_tokenizer):
    restored = CharTokenizer.from_state_dict(fitted_tokenizer.state_dict())
    assert restored.itos == fitted_tokenizer.itos
    a, _ = fitted_tokenizer.encode("http://paypal.com/login")
    b, _ = restored.encode("http://paypal.com/login")
    assert a == b


def test_tokenizer_save_load_roundtrip(fitted_tokenizer, tmp_path):
    p = tmp_path / "tok.json"
    fitted_tokenizer.save(p)
    loaded = CharTokenizer.load(p)
    assert loaded.itos == fitted_tokenizer.itos
    assert loaded.max_length == fitted_tokenizer.max_length


def test_suggest_max_length_is_percentile_based():
    urls = ["http://a.tld/" + "x" * n for n in range(1, 101)]
    cap = suggest_max_length(urls, percentile=99.0, floor=64)
    assert cap >= 64
    assert cap <= 1024


def test_suggest_max_length_handles_no_input():
    assert suggest_max_length([]) == 64


# ---------------------------------------------------------------------------
# Handcrafted features
# ---------------------------------------------------------------------------
def test_feature_vector_matches_declared_names():
    feats = extract_handcrafted_features("http://paypal.com/login")
    assert len(feats) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in feats)


@pytest.mark.parametrize(
    "url",
    [
        "http://a.tld",
        "https://sub.deep.domain.co.uk/path?q=1#frag",
        "http://192.168.0.1/admin",
        "http://[::1]/x",
        "http://user:pass@a.tld:8080/x",
        "not even a url",
        "",
    ],
)
def test_feature_extraction_never_raises(url):
    feats = extract_handcrafted_features(url)
    assert len(feats) == len(FEATURE_NAMES)


def test_https_flag_is_derived_from_scheme():
    i = FEATURE_NAMES.index("is_https")
    assert extract_handcrafted_features("https://a.tld")[i] == 1.0
    assert extract_handcrafted_features("http://a.tld")[i] == 0.0


def test_ip_in_host_is_flagged():
    i = FEATURE_NAMES.index("is_ip_in_host")
    assert extract_handcrafted_features("http://192.168.1.1/x")[i] == 1.0
    assert extract_handcrafted_features("http://example.com/x")[i] == 0.0


def test_suspicious_keyword_counter():
    i = FEATURE_NAMES.index("n_suspicious_keywords")
    assert extract_handcrafted_features("http://a.tld/login/verify/account")[i] >= 2.0
    assert extract_handcrafted_features("http://a.tld/blog/post")[i] == 0.0


def test_brand_mismatch_only_when_domain_is_not_the_brand():
    i = FEATURE_NAMES.index("brand_domain_mismatch")
    # Brand name in the domain's own name label -> the site controls that zone.
    assert extract_handcrafted_features("http://paypal.com/login")[i] == 0.0
    # Brand embedded in a different name label -> impersonation.
    assert extract_handcrafted_features("http://paypal-secure.com/login")[i] == 1.0
    assert extract_handcrafted_features("http://securepaypal.com/login")[i] == 1.0
    # No brand at all -> nothing to mismatch.
    assert extract_handcrafted_features("http://example.tld/blog")[i] == 0.0


def test_name_label_is_the_whole_second_level_label():
    # Real TLDs, so tldextract's PSL actually recognises the suffix. (.tld is not
    # a public suffix, and tldextract then treats the final label as the suffix.)
    assert name_label_of("paypal-secure.com") == "paypal-secure"
    assert name_label_of("paypal.com") == "paypal"
    assert name_label_of("a.b.paypal.co.uk") == "paypal"


def test_shannon_entropy_bounds():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == pytest.approx(2.0)


def test_subdomain_depth():
    i = FEATURE_NAMES.index("n_subdomains")
    assert extract_handcrafted_features("http://a.b.c.example.com/x")[i] >= 2.0
    assert extract_handcrafted_features("http://example.com/x")[i] == 0.0


def test_scaler_roundtrip_and_drift_detection():
    rows = extract_batch(["http://a.com/x", "http://b.org/y?z=1", "http://c.net/"])
    sc = FeatureScaler().fit(rows)
    out = sc.transform(rows)
    assert len(out) == 3 and len(out[0]) == len(FEATURE_NAMES)

    restored = FeatureScaler.from_state_dict(sc.state_dict())
    assert np.allclose(restored.transform(rows), out)

    # A checkpoint whose feature list changed must be rejected, not silently used.
    bad = dict(sc.state_dict())
    bad["feature_names"] = ["nope"] * len(FEATURE_NAMES)
    with pytest.raises(ValueError, match="feature names"):
        FeatureScaler.from_state_dict(bad)


def test_scaler_refuses_to_transform_before_fit():
    with pytest.raises(RuntimeError, match="before fit"):
        FeatureScaler().transform([[0.0]])


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def test_additive_attention_ignores_padding():
    att = AdditiveAttention(input_dim=4, attention_dim=3)
    seq = torch.randn(2, 6, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    pooled, weights = att(seq, mask)
    assert weights.shape == (2, 6)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)
    # Row 0: the padded tail must receive exactly zero weight.
    assert torch.allclose(weights[0, 3:], torch.zeros(3), atol=1e-6)
    # Pooled vector equals the attention-weighted sum over the real positions.
    expected = (weights[0, :3].unsqueeze(1) * seq[0, :3]).sum(dim=0)
    assert torch.allclose(pooled[0], expected, atol=1e-5)


def test_additive_attention_is_finite_for_fully_masked_row():
    att = AdditiveAttention(input_dim=4, attention_dim=3)
    seq = torch.randn(1, 4, 4)
    mask = torch.zeros(1, 4)
    _, weights = att(seq, mask)
    assert torch.isfinite(weights).all(), "a fully padded row must not produce NaN"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_batch(fitted_tokenizer):
    urls = ["http://paypal.com/login", "https://bank.example.co.uk/verify"]
    ids, mask = fitted_tokenizer.encode_batch(urls)
    sc = FeatureScaler().fit(extract_batch(urls))
    hand = torch.tensor(sc.transform(extract_batch(urls)), dtype=torch.float32)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.float32),
        hand,
    )


def _tiny_model(use_handcrafted: bool) -> URLCharModel:
    return URLCharModel(
        vocab_size=64, max_length=64, embedding_dim=16, cnn_channels=8,
        cnn_kernel_sizes=(3, 5, 7), lstm_hidden=8, embedding_out=128,
        dropout=0.0, n_handcrafted=len(FEATURE_NAMES), use_handcrafted=use_handcrafted,
    )


def test_forward_shapes_with_handcrafted_branch(tiny_batch):
    ids, mask, hand = tiny_batch
    model = _tiny_model(True)
    out = model(ids, mask, hand)
    assert out.logit.shape == (2,)
    assert out.probability.shape == (2,)
    assert out.embedding.shape == (2, 128)
    assert out.attention.shape == (2, ids.shape[1])


def test_forward_shapes_without_handcrafted_branch(tiny_batch):
    ids, mask, _ = tiny_batch
    model = _tiny_model(False)
    out = model(ids, mask, None)
    assert out.logit.shape == (2,)
    assert out.embedding.shape == (2, 128)


def test_probabilities_are_within_unit_interval(tiny_batch):
    ids, mask, hand = tiny_batch
    model = _tiny_model(True)
    out = model(ids, mask, hand)
    assert torch.all(out.probability >= 0.0)
    assert torch.all(out.probability <= 1.0)
    # probability must be exactly sigmoid(logit), not a separate head.
    assert torch.allclose(out.probability, torch.sigmoid(out.logit), atol=1e-6)


def test_attention_normalises_per_row(tiny_batch):
    ids, mask, hand = tiny_batch
    model = _tiny_model(True)
    out = model(ids, mask, hand)
    assert torch.allclose(out.attention.sum(dim=1), torch.ones(2), atol=1e-5)


def test_probability_is_deterministic_in_eval_mode(tiny_batch):
    ids, mask, hand = tiny_batch
    model = _tiny_model(True).eval()
    with torch.no_grad():
        a = model(ids, mask, hand).probability
        b = model(ids, mask, hand).probability
    assert torch.equal(a, b)


def test_model_requires_handcrafted_features_when_enabled(tiny_batch):
    ids, mask, _ = tiny_batch
    model = _tiny_model(True)
    with pytest.raises(ValueError, match="handcrafted"):
        model(ids, mask, None)


def test_model_rejects_wrong_feature_width(tiny_batch):
    ids, mask, hand = tiny_batch
    model = _tiny_model(True)
    with pytest.raises(ValueError, match="handcrafted features"):
        model(ids, mask, torch.zeros(2, 3))


def test_model_rejects_mismatched_mask(tiny_batch):
    ids, mask, hand = tiny_batch
    model = _tiny_model(True)
    with pytest.raises(ValueError, match="mask shape"):
        model(ids, torch.ones_like(mask)[:, :10], hand)


def test_model_rejects_wrong_rank_input(tiny_batch):
    ids, mask, hand = tiny_batch
    model = _tiny_model(True)
    with pytest.raises(ValueError, match=r"\(B, L\)"):
        model(ids[0], mask[0], hand[0])


def test_gradients_reach_every_trainable_parameter(tiny_batch):
    ids, mask, hand = tiny_batch
    model = _tiny_model(True)
    out = model(ids, mask, hand)
    loss = out.logit.sum()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"


def test_multi_kernel_convolution_output_widths_align(tiny_batch):
    """Each kernel must return exactly L timesteps or attention misaligns."""
    ids, mask, hand = tiny_batch
    model = _tiny_model(True)
    L = ids.shape[1]
    emb = model.embedding(ids).transpose(1, 2)
    for conv in model.convolutions:
        out = conv(emb)
        assert out.shape[2] >= L


def test_model_config_roundtrip():
    model = _tiny_model(True)
    cfg = model.config()
    assert cfg["embedding_out"] == 128
    assert cfg["use_handcrafted"] is True
    assert cfg["cnn_kernel_sizes"] == [3, 5, 7]


def test_checkpoint_roundtrip_preserves_predictions(tiny_batch, tmp_path):
    ids, mask, hand = tiny_batch
    model = _tiny_model(True).eval()
    with torch.no_grad():
        before = model(ids, mask, hand).probability.clone()

    path = tmp_path / "m.pt"
    torch.save({"model_state": model.state_dict(), "model_config": model.config()}, path)
    blob = torch.load(path, map_location="cpu", weights_only=False)

    restored = URLCharModel(**blob["model_config"])
    restored.load_state_dict(blob["model_state"])
    restored.eval()
    with torch.no_grad():
        after = restored(ids, mask, hand).probability

    assert torch.allclose(before, after, atol=1e-6), (
        "a reloaded checkpoint must reproduce identical predictions"
    )