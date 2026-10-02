"""Tests for the HTML, vision and fusion models.

The fusion tests concentrate on the mask behaviour, because that is where a
silent correctness bug would be invisible in the metrics: a gate that ignores
availability still produces plausible-looking probabilities.
"""

from __future__ import annotations

import pytest
import torch

from app.models.fusion_model import FusionModel
from app.models.html_model import HTMLModalityModel
from app.preprocessing.html_features import (
    HTML_FEATURE_NAMES,
    N_HTML_FEATURES,
    extract_html_features,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# HTML feature extractor
# ---------------------------------------------------------------------------
def test_feature_vector_has_the_declared_width():
    assert len(extract_html_features("<html><body>hi</body></html>")) == N_HTML_FEATURES
    assert len(HTML_FEATURE_NAMES) == N_HTML_FEATURES
    assert len(set(HTML_FEATURE_NAMES)) == N_HTML_FEATURES


def test_missing_document_yields_all_zeros():
    assert set(extract_html_features(None)) == {0.0}
    assert set(extract_html_features("")) == {0.0}


@pytest.mark.parametrize("bad", ["<html", "<<<>>>", "\x00\x01", "<a href=>"])
def test_malformed_markup_never_raises(bad):
    assert len(extract_html_features(bad)) == N_HTML_FEATURES


def test_credentials_page_scores_higher_than_a_plain_page():
    phish = """<html><head><title>PayPal Secure Login</title></head><body>
    <form action="https://evil.test/steal">
    <input type="password" name="password"><input type="text" name="username">
    <input name="cardnumber"></form><iframe src="//ads.test"></iframe></body></html>"""
    legit = "<html><head><title>Example</title></head><body><p>Hello world.</p></body></html>"
    a, b = extract_html_features(phish), extract_html_features(legit)
    assert a[4] > b[4], "password field count"
    assert a[5] > b[5], "form count"
    assert a[7] > b[7], "iframe count"
    assert a[24] > b[24], "brand/host mismatch"


def test_script_and_style_text_is_excluded_from_visible_text():
    html = "<html><body><script>var secret='hunter2';</script><p>real text</p></body></html>"
    from app.preprocessing.html_features import visible_text_of

    from app.preprocessing.multimodal_dataset import _soup_of

    text = visible_text_of(_soup_of(html))
    assert "real text" in text
    assert "hunter2" not in text


# ---------------------------------------------------------------------------
# HTML model
# ---------------------------------------------------------------------------
def _html_model():
    return HTMLModalityModel(
        vocab_size=64, max_tokens=32, n_dom_features=N_HTML_FEATURES,
        embedding_dim=16, transformer_layers=1, transformer_heads=2,
        transformer_ff=32, embedding_out=16,
    )


def test_html_model_produces_an_embedding_and_a_logit():
    m = _html_model()
    ids = torch.randint(1, 64, (3, 32))
    mask = torch.ones(3, 32)
    out = m(ids, mask, torch.randn(3, N_HTML_FEATURES))
    assert out.embedding.shape == (3, 16)
    assert out.logit.shape == (3,)


def test_html_model_handles_absent_text_branch():
    m = _html_model()
    out = m(torch.zeros(0, dtype=torch.long), torch.zeros(0), torch.randn(2, N_HTML_FEATURES))
    assert out.embedding.shape == (2, 16)
    assert torch.isfinite(out.logit).all()


def test_html_model_survives_a_fully_masked_row():
    """All-padding would make softmax produce NaN if the mask leaked through."""
    m = _html_model()
    ids = torch.randint(1, 64, (2, 32))
    mask = torch.zeros(2, 32)
    out = m(ids, mask, torch.randn(2, N_HTML_FEATURES))
    assert torch.isfinite(out.logit).all()


def test_html_model_gradients_flow():
    m = _html_model()
    ids = torch.randint(1, 64, (2, 32))
    out = m(ids, torch.ones(2, 32), torch.randn(2, N_HTML_FEATURES))
    out.logit.sum().backward()
    assert m.token_emb.weight.grad is not None


# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------
def _fusion():
    # eval() so dropout cannot make two forward passes differ; these tests
    # compare outputs and want the mask to be the only thing that matters.
    return FusionModel(
        shared_dim=16, hidden_dims=(16, 8), modality_dropout=0.0
    ).eval()


def _inputs(n=4):
    return (
        torch.randn(n, 16), torch.randn(n, 16), torch.randn(n, 16),
        torch.ones(n), torch.ones(n), torch.ones(n),
    )


def test_fusion_returns_a_logit_and_gates():
    f = _fusion()
    out = f(*_inputs())
    assert out.logit.shape == (4,)
    assert out.gates.shape == (4, 3)
    assert set(out.modality_logits) == {"url", "html", "vision"}


def test_gates_are_a_probability_distribution():
    out = _fusion()(*_inputs())
    assert torch.allclose(out.gates.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (out.gates >= 0).all()


def test_missing_modality_receives_exactly_zero_weight():
    """The whole point of the mask: an absent branch must not be blended in."""
    f = _fusion()
    eu, eh, ev, mu, mh, mv = _inputs()
    out = f(eu, eh, ev, mu, mh, torch.zeros_like(mv))
    assert torch.allclose(out.gates[:, 2], torch.zeros(4), atol=1e-6)


def test_available_modality_weights_still_sum_to_one():
    f = _fusion()
    eu, eh, ev, mu, mh, mv = _inputs()
    out = f(eu, eh, ev, mu, torch.zeros_like(mh), mv)
    assert torch.allclose(out.gates.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(out.gates[:, 1], torch.zeros(4), atol=1e-6)


def test_url_only_row_puts_all_weight_on_the_url():
    f = _fusion()
    eu, eh, ev, mu, mh, mv = _inputs()
    out = f(eu, eh, ev, mu, torch.zeros_like(mh), torch.zeros_like(mv))
    assert torch.allclose(out.gates[:, 0], torch.ones(4), atol=1e-6)


def test_a_row_with_no_modality_still_produces_a_finite_logit():
    """Degenerate input must not produce NaN; it falls back to the URL slot."""
    f = _fusion()
    z = torch.zeros(4)
    out = f(torch.randn(4, 16), torch.randn(4, 16), torch.randn(4, 16), z, z, z)
    assert torch.isfinite(out.logit).all()


def test_changing_a_present_modality_changes_the_output():
    """Guards against a gate that is computed but then ignored."""
    f = _fusion()
    eu, eh, ev, mu, mh, mv = _inputs()
    base = f(eu, eh, ev, mu, mh, mv).logit
    other = f(eu, eh + 5.0, ev, mu, mh, mv).logit
    assert not torch.allclose(base, other, atol=1e-4)


def test_changing_an_absent_modality_does_not_change_the_output():
    """Content behind a zero mask must be invisible to the model."""
    f = _fusion()
    eu, eh, ev, mu, mh, mv = _inputs()
    zero = torch.zeros_like(mh)
    a = f(eu, eh, ev, mu, zero, mv).logit
    b = f(eu, eh + 5.0, ev, mu, zero, mv).logit
    assert torch.allclose(a, b, atol=1e-5)


def test_fusion_is_differentiable():
    f = _fusion()
    out = f(*_inputs())
    out.logit.sum().backward()
    assert f.head.weight.grad is not None
    assert f.gate[0].weight.grad is not None


def test_fusion_batch_size_one_works():
    f = _fusion()
    out = f(*_inputs(1))
    assert out.gates.shape == (1, 3)
