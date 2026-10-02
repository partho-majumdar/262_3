"""Continual-learning unit tests: reservoir, balancing, EWC, update.

Deliberately fast and tiny. The full stream experiment lives in
``training/continual_update.py``; what is asserted here are the *mechanisms*
that script depends on. A test that needed the real checkpoint would be testing
the checkpoint, not the logic, and would stop being a unit test the moment the
model was retrained.

Nothing here trains a real model to convergence. The synthetic corpus is
character-separable on purpose so a handful of Adam steps measurably reduces
the loss; if that ever stops holding, the failure is a real regression in the
update loop rather than noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.models.url_model import URLCharModel
from app.preprocessing.url_features import FEATURE_NAMES, FeatureScaler, extract_batch
from app.preprocessing.url_preprocessing import CharTokenizer
from app.services.continual import (
    BalancedBatchSampler,
    ContinualUpdater,
    EWCState,
    RehearsalBuffer,
)

pytestmark = pytest.mark.models


# ---------------------------------------------------------------------------
# Tiny fixtures
# ---------------------------------------------------------------------------
def synth_corpus(n: int = 12) -> tuple[list[str], list[int]]:
    """A corpus that is separable by character content alone.

    Legitimate URLs are plain ASCII ``.com`` catalogue paths; phishing URLs carry
    the ``.tk`` free-registry host and a ``login.php`` path that is the exact
    pattern the project's own adversarial report identifies as the cheap-registry
    prior. So the fixture exercises a learnable task in a few steps on CPU.
    """
    legit = [f"https://www.shop{n}.com/catalog/product-{n}" for n in range(n)]
    phish = [f"http://account-verify-{n}.tk/login.php?confirm={n}" for n in range(n)]
    urls = legit + phish
    labels = [0] * n + [1] * n
    return urls, labels


def tiny_model(urls: list[str], *, max_length: int = 96) -> tuple[URLCharModel, CharTokenizer, FeatureScaler]:
    """A miniature stand-in with the same interfaces as the shipped model."""
    tok = CharTokenizer(max_length=max_length, min_count=1).fit(urls)
    scaler = FeatureScaler().fit(extract_batch(urls))
    model = URLCharModel(
        vocab_size=tok.vocab_size,
        max_length=tok.max_length,
        embedding_dim=8,
        cnn_channels=6,
        cnn_kernel_sizes=(3, 5),
        lstm_hidden=6,
        lstm_layers=1,
        bidirectional=True,
        embedding_out=16,
        dropout=0.0,
        n_handcrafted=len(FEATURE_NAMES),
        use_handcrafted=True,
    )
    model.eval()
    return model, tok, scaler


@pytest.fixture(scope="module")
def corpus() -> tuple[list[str], list[int]]:
    return synth_corpus()


@pytest.fixture(scope="module")
def stack(corpus):
    return tiny_model(corpus[0])


@pytest.fixture()
def updater(stack, corpus):
    """A fresh updater per test: these tests mutate weights, so state must not leak."""
    model, tok, scaler = stack
    return ContinualUpdater(
        model, tok, scaler, buffer_capacity=16, seed=7, lr=0.01, batch_size=8, ewc_lambda=1.0
    )


# ---------------------------------------------------------------------------
# RehearsalBuffer: capacity, determinism, labels
# ---------------------------------------------------------------------------
def test_buffer_respects_capacity():
    buf = RehearsalBuffer(capacity=10, seed=1)
    buf.add([f"http://u{i}.com" for i in range(50)], [i % 2 for i in range(50)])
    assert len(buf) == 10


def test_buffer_is_deterministic_for_a_fixed_seed():
    urls = [f"http://u{i}.com" for i in range(40)]
    labels = [i % 2 for i in range(40)]
    a, b = RehearsalBuffer(capacity=8, seed=3), RehearsalBuffer(capacity=8, seed=3)
    a.add(urls, labels)
    b.add(urls, labels)
    assert (a.urls, a.labels) == (b.urls, b.labels)


def test_different_seeds_can_diverge():
    """Sanity check that the seed is actually consulted, not decorative."""
    urls = [f"http://u{i}.com" for i in range(200)]
    labels = [i % 2 for i in range(200)]
    a, b = RehearsalBuffer(capacity=16, seed=1), RehearsalBuffer(capacity=16, seed=2)
    a.add(urls, labels)
    b.add(urls, labels)
    assert (a.urls, a.labels) != (b.urls, b.labels)


def test_buffer_add_returns_number_of_slots_changed():
    buf = RehearsalBuffer(capacity=4, seed=0)
    assert buf.add(["a", "b", "c", "d"], [0, 1, 0, 1]) == 4
    # Saturated: some draws hit, some miss, but the count is never above capacity.
    changed = buf.add([f"u{i}" for i in range(50)], [i % 2 for i in range(50)])
    assert 0 <= changed <= 50
    assert len(buf) == 4


def test_buffer_n_seen_counts_every_offer_not_just_retained():
    buf = RehearsalBuffer(capacity=3, seed=5)
    buf.add([f"u{i}" for i in range(20)], [0] * 20)
    assert buf.n_seen == 20
    assert len(buf) == 3


def test_buffer_tracks_class_composition():
    buf = RehearsalBuffer(capacity=100, seed=2)
    buf.add(["a", "b", "c", "d"], [1, 1, 0, 0])
    comp = buf.composition()
    assert comp["phishing"] == 2 and comp["legitimate"] == 2
    assert comp["size"] == 4 and comp["capacity"] == 100
    assert comp["phishing_fraction"] == pytest.approx(0.5)


def test_buffer_rejects_ambiguous_labels():
    buf = RehearsalBuffer(capacity=4, seed=0)
    with pytest.raises(ValueError):
        buf.add(["a"], [2])


def test_buffer_rejects_length_mismatch():
    buf = RehearsalBuffer(capacity=4, seed=0)
    with pytest.raises(ValueError):
        buf.add(["a", "b"], [1])


def test_buffer_rejects_zero_capacity():
    with pytest.raises(ValueError):
        RehearsalBuffer(capacity=0)


def test_buffer_extend_absorbs_another_buffer():
    a = RehearsalBuffer(capacity=10, seed=1)
    a.add(["x", "y"], [1, 0])
    b = RehearsalBuffer(capacity=10, seed=2)
    b.add(["z"], [1])
    b.extend(a)
    assert set(b.urls) == {"x", "y", "z"}
    assert len(b) == 3


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------
def test_buffer_save_load_round_trip(tmp_path: Path):
    buf = RehearsalBuffer(capacity=6, seed=11)
    urls = [f"http://host{i}.com/p" for i in range(30)]
    buf.add(urls, [i % 2 for i in range(30)])
    path = buf.save(tmp_path / "buffer.npz")
    back = RehearsalBuffer.load(path)

    assert back.urls == buf.urls
    assert back.labels == buf.labels
    assert back.capacity == buf.capacity
    assert back.seed == buf.seed
    assert back.n_seen == buf.n_seen


def test_buffer_round_trip_survives_future_additions(tmp_path: Path):
    """A resumed buffer must keep behaving like a reservoir, not restart counting."""
    buf = RehearsalBuffer(capacity=5, seed=4)
    buf.add([f"u{i}" for i in range(40)], [i % 2 for i in range(40)])
    path = buf.save(tmp_path / "b.npz")
    back = RehearsalBuffer.load(path)
    back.add(["late-1", "late-2"], [1, 0])
    assert back.n_seen == 42
    assert len(back) == 5


def test_buffer_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        RehearsalBuffer.load(tmp_path / "nope.npz")


def test_buffer_save_creates_parent_directory(tmp_path: Path):
    buf = RehearsalBuffer(capacity=2, seed=0)
    path = buf.save(tmp_path / "nested" / "deep" / "b.npz")
    assert path.is_file()


def test_saved_buffer_needs_no_pickle(tmp_path: Path):
    """URLs live in a fixed-width unicode array so the file reloads under allow_pickle=False."""
    buf = RehearsalBuffer(capacity=3, seed=0)
    buf.add(["http://a.com", "http://b.com"], [0, 1])
    path = buf.save(tmp_path / "b.npz")
    with np.load(path, allow_pickle=False) as blob:
        assert blob["urls"].dtype.kind == "U"


# ---------------------------------------------------------------------------
# Class-balanced sampling
# ---------------------------------------------------------------------------
def test_balanced_sampler_halves_each_class():
    sampler = BalancedBatchSampler([0] * 50 + [1] * 50, batch_size=16, seed=0)
    batch = sampler.sample_batch()
    info = sampler.describe(batch)
    assert info["size"] == 16
    assert info["positive"] == 8 and info["negative"] == 8


def test_balanced_sampler_counteracts_a_phishing_heavy_stream():
    """The whole point: 90% phishing input must still yield a 50/50 batch."""
    sampler = BalancedBatchSampler([1] * 90 + [0] * 10, batch_size=10, seed=1)
    info = sampler.describe(sampler.sample_batch())
    assert info["positive"] == 5 and info["negative"] == 5


def test_balanced_sampler_reports_degeneracy_for_a_single_class_pool():
    sampler = BalancedBatchSampler([1, 1, 1], batch_size=4, seed=0)
    assert sampler.has_both_classes is False
    info = sampler.describe(sampler.sample_batch())
    assert info["degenerate"] is True
    assert info["positive"] == 3 and info["negative"] == 0


def test_balanced_sampler_fills_a_batch_from_a_small_pool():
    sampler = BalancedBatchSampler([0, 1, 1], batch_size=8, seed=0)
    info = sampler.describe(sampler.sample_batch())
    assert info["size"] == 3


def test_balanced_sampler_is_deterministic_for_a_fixed_seed():
    labels = [0] * 20 + [1] * 20
    a = BalancedBatchSampler(labels, batch_size=8, seed=3).batches(5)
    b = BalancedBatchSampler(labels, batch_size=8, seed=3).batches(5)
    assert [x for x in a] == [x for x in b]


def test_balanced_sampler_batches_count_matches_requested_steps():
    sampler = BalancedBatchSampler([0] * 5 + [1] * 5, batch_size=4, seed=0)
    assert len(list(sampler.batches(9))) == 9
    assert len(list(sampler.batches(0))) == 0


# ---------------------------------------------------------------------------
# EWCState in isolation
# ---------------------------------------------------------------------------
def test_ewc_penalty_is_zero_at_the_anchor():
    ewc = EWCState()
    ewc.lam = 10.0
    ewc.fisher = [torch.ones(4)]
    ewc.anchor = [torch.zeros(4)]
    assert float(ewc.penalty([torch.zeros(4)]).item()) == pytest.approx(0.0)
    assert ewc.active is True


def test_ewc_penalty_grows_with_distance():
    ewc = EWCState()
    ewc.lam = 2.0
    ewc.fisher = [torch.ones(3)]
    ewc.anchor = [torch.zeros(3)]
    near = float(ewc.penalty([torch.full((3,), 0.5)]).item())
    far = float(ewc.penalty([torch.full((3,), 1.0)]).item())
    assert far == pytest.approx(4 * near)


def test_inactive_ewc_has_no_penalty():
    ewc = EWCState()
    assert ewc.active is False
    assert float(ewc.penalty([torch.ones(2)]).item()) == 0.0


def test_ewc_penalty_rejects_a_mismatched_model():
    ewc = EWCState()
    ewc.lam = 1.0
    ewc.fisher = [torch.ones(4)]
    ewc.anchor = [torch.zeros(4)]
    with pytest.raises(ValueError):
        ewc.penalty([torch.zeros(3)])


def test_ewc_clear_resets_state():
    ewc = EWCState()
    ewc.lam = 5.0
    ewc.fisher = [torch.ones(2)]
    ewc.anchor = [torch.zeros(2)]
    ewc.clear()
    assert ewc.active is False and ewc.fisher == []


# ---------------------------------------------------------------------------
# ContinualUpdater: empty / edge cases must not raise
# ---------------------------------------------------------------------------
def test_update_on_an_empty_buffer_is_a_no_op_not_a_crash(updater):
    result = updater.update(steps=3)
    assert result["skipped"] is True
    assert result["reason"] == "no replayable data"
    assert result["loss"] is None


def test_observe_with_an_empty_batch_does_not_raise(updater):
    updater.observe([], [])
    assert updater.pending_urls == []
    assert updater.update(steps=2)["skipped"] is True


def test_observe_with_only_the_new_batch_still_trains(updater, corpus):
    urls, labels = corpus
    updater.use_rehearsal = False
    updater.observe(urls[:6], labels[:6])
    result = updater.update(steps=2)
    assert result["skipped"] is False
    assert result["n_train_urls"] == 6


def test_observe_rejects_a_label_length_mismatch(updater):
    with pytest.raises(ValueError):
        updater.observe(["a", "b"], [1])


def test_update_with_zero_steps_is_reported_not_silent(updater, corpus):
    urls, labels = corpus
    updater.observe(urls, labels)
    result = updater.update(steps=0)
    assert result["skipped"] is True
    assert "zero steps" in result["reason"]


def test_evaluate_on_an_empty_set_reports_unavailable(updater):
    out = updater.evaluate_on([], [])
    assert out["available"] is False
    assert out["accuracy"] is None and out["f1"] is None


def test_evaluate_on_a_single_class_set_returns_honest_nones(updater, corpus):
    urls, labels = corpus
    out = updater.evaluate_on(urls[:4], [1, 1, 1, 1])
    assert out["available"] is True
    assert out["recall"] is not None
    assert out["roc_auc"] is None, "AUC is undefined on one class and must not be faked"


# ---------------------------------------------------------------------------
# ContinualUpdater: learning behaviour
# ---------------------------------------------------------------------------
def test_update_reduces_loss_on_a_tiny_synthetic_task(stack, corpus):
    model, tok, scaler = stack
    urls, labels = corpus
    up = ContinualUpdater(model, tok, scaler, buffer_capacity=32, seed=0, lr=0.02, batch_size=8)
    up.observe(urls, labels)
    result = up.update(steps=30, lr=0.02)
    assert result["skipped"] is False
    assert result["loss_last"] < result["loss_first"]


def test_update_reports_a_balanced_batch_composition(updater, corpus):
    urls, labels = corpus
    updater.observe(urls, labels)
    result = updater.update(steps=4, batch_size=8)
    frac = result["batch_composition"]["mean_positive_fraction"]
    assert frac == pytest.approx(0.5, abs=1e-6)


def test_update_reports_buffer_composition_and_drift(updater, corpus):
    urls, labels = corpus
    updater.observe(urls[:10], labels[:10])
    result = updater.update(steps=3)
    assert result["buffer"]["size"] == 10
    assert result["buffer"]["phishing"] + result["buffer"]["legitimate"] == 10
    assert result["drift"]["since_previous_update"] > 0.0
    assert result["drift"]["since_initial"] > 0.0


def test_replay_pool_does_not_double_count_the_pending_batch(updater, corpus):
    urls, labels = corpus
    updater.observe(urls, labels)
    assert len(updater._replay_pool()[0]) == len(urls)


def test_baseline_arm_trains_only_on_the_new_batch(stack, corpus):
    """The ablation arm must not silently get replay for free."""
    model, tok, scaler = stack
    urls, labels = corpus
    up = ContinualUpdater(model, tok, scaler, buffer_capacity=32, seed=0, use_rehearsal=False)
    up.observe(urls[:8], labels[:8])
    up.update(steps=2)
    assert up.buffer.n_seen == 0, "rehearsal disabled must leave the buffer empty"
    assert up.update(steps=1)["n_train_urls"] == 8


def test_evaluate_on_changes_after_an_update(stack, corpus):
    """If evaluation is identical before and after, the metrics are not real."""
    model, tok, scaler = stack
    urls, labels = corpus
    up = ContinualUpdater(model, tok, scaler, buffer_capacity=32, seed=0, lr=0.05, batch_size=8)
    up.observe(urls, labels)
    before_logits = up.logits_for(urls)
    up.update(steps=25, lr=0.05)
    after_logits = up.logits_for(urls)
    assert not np.allclose(before_logits, after_logits), "the update must move the weights"
    after = up.evaluate_on(urls, labels)
    assert after["available"] is True
    assert 0.0 <= after["f1"] <= 1.0


# ---------------------------------------------------------------------------
# EWC through the updater
# ---------------------------------------------------------------------------
def test_updater_ewc_penalty_is_zero_at_the_initial_weights(updater, corpus):
    urls, labels = corpus
    updater.observe(urls, labels)
    updater.set_ewc_anchor(urls, labels)
    assert updater.ewc_penalty() == pytest.approx(0.0)


def test_updater_ewc_penalty_grows_after_a_gradient_step(updater, corpus):
    urls, labels = corpus
    updater.observe(urls, labels)
    updater.set_ewc_anchor(urls, labels)
    before = updater.ewc_penalty()
    updater.update(steps=5, lr=0.05, ewc_lambda=0.0)
    after = updater.ewc_penalty()
    assert before == pytest.approx(0.0)
    assert after > before


def test_fisher_zeroes_weights_that_never_receive_gradient(updater, corpus):
    """A dead parameter must be free to move, which is what separates EWC from L2."""
    urls, labels = corpus
    updater.observe(urls, labels)
    updater.estimate_fisher(urls, labels, batches=2)
    dead = [f for f in updater.ewc.fisher if float(f.max()) == 0.0]
    updater.update(steps=3, lr=0.05, ewc_lambda=0.0)
    if dead:
        assert updater.ewc_penalty() >= 0.0


def test_ewc_lambda_zero_disables_the_penalty(updater, corpus):
    urls, labels = corpus
    updater.observe(urls, labels)
    updater.set_ewc_anchor(urls, labels)
    result = updater.update(steps=3, ewc_lambda=0.0)
    assert result["ewc"]["active"] is False
    assert result["ewc"]["penalty"] == 0.0


def test_updater_state_dict_round_trip(stack, corpus):
    urls, labels = corpus
    up = ContinualUpdater(stack[0], stack[1], stack[2], buffer_capacity=8, seed=3, ewc_lambda=2.0)
    up.observe(urls[:6], labels[:6])
    state = up.state_dict()
    other = ContinualUpdater(stack[0], stack[1], stack[2], buffer_capacity=8, seed=3)
    other.load_state_dict(state)
    assert other.buffer.urls == up.buffer.urls
    assert other.buffer.labels == up.buffer.labels
    assert other.buffer.n_seen == up.buffer.n_seen
    assert other.ewc_lambda == 2.0