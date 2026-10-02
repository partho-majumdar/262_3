"""Acceptance tests for the domain-IP graph builder and the GCN.

No training here: these are fast shape / invariant / determinism checks that can
run in a fraction of a second, so a broken graph invariant fails the unit suite
rather than showing up three minutes into a training run.

The important one is
:func:`test_edge_set_identical_under_label_permutation`. If any future edit ever
lets a label influence an edge, that test fails immediately - which is the whole
point of building the graph in a separate, testable module.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from app.models.graph_model import GCNOutput, GraphConv, PhishingGCN
from app.preprocessing.graph_build import (
    EDGE_CONTAINMENT,
    EDGE_DOMAIN_IP,
    EDGE_DOMAIN_SUBNET,
    EDGE_IP_SUBNET,
    EDGE_SHARED_IP,
    EDGE_SHARED_SUBDOMAIN,
    EDGE_SHARED_TLD,
    GRAPH_FEATURE_NAMES,
    N_EDGE_TYPES,
    N_FEATURES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    GraphSpec,
    build_graph,
    load_graph,
    node_features,
    save_graph,
    subnet_of,
)
from training.train_gnn import verify_split_grouping

BACKEND_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
#: A handcrafted corpus small enough that every node and edge can be counted by
#: hand. The structure is deliberately asymmetric so a wrong grouping (e.g. using
#: the full hostname instead of the registrable domain) produces different counts.
#: The TLDs are **real** public suffixes on purpose: for an unlisted suffix,
#: ``tldextract`` returns an empty ``registered_domain`` and both
#: ``make_splits.registered_domain_of`` and this module fall back to the whole
#: hostname - which would quietly make "one node per registrable domain" false.
TINY_URLS: list[str] = [
    "http://login.secure-bank-a.com/login",
    "http://login.secure-bank-a.com/verify",
    "http://shop.secure-bank-b.com/cart",
    "http://secure-bank-c.com/account",
    "http://www.museum-site.org/index",
    "http://www.museum-site.org/about",
    "http://deep.sub.unknown-hosting.xyz/ping",
    "http://198.51.100.7/login",
    "http://203.0.113.9/admin",
]
TINY_LABELS: list[int] = [1, 1, 1, 1, 0, 0, 0, 1, 0]
TINY_SPLITS: list[str] = [
    "train", "train", "train", "train",
    "val", "val", "val",
    "test", "test",
]


@pytest.fixture()
def tiny_graph() -> GraphSpec:
    return build_graph(TINY_URLS, TINY_LABELS, splits=TINY_SPLITS)


@pytest.fixture()
def tiny_resolved_graph() -> GraphSpec:
    """The same corpus with simulated DNS answers - the only way IP edges appear."""
    return build_graph(
        TINY_URLS,
        TINY_LABELS,
        extra_domains={
            "secure-bank-a.com": ["198.51.100.7"],
            "secure-bank-b.com": ["198.51.100.7"],
            "museum-site.org": ["192.0.2.10"],
        },
        splits=TINY_SPLITS,
    )


# ---------------------------------------------------------------------------
# node_features
# ---------------------------------------------------------------------------
def test_node_features_length_matches_declared_names():
    vec = node_features("example.com")
    assert len(vec) == N_FEATURES
    assert len(GRAPH_FEATURE_NAMES) == N_FEATURES
    assert len(set(GRAPH_FEATURE_NAMES)) == N_FEATURES


def test_node_features_is_deterministic_across_calls():
    """Same string, same numbers, twice - the baseline requirement."""
    a = node_features("paypal-secure-login.tld")
    b = node_features("paypal-secure-login.tld")
    assert a == b


def test_node_features_is_deterministic_across_processes():
    """Python's ``hash()`` is salted per process; this catches a regression to it.

    Run in a **separate interpreter** so the hash seed genuinely differs from
    this process's.
    """
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from app.preprocessing.graph_build import node_features;"
        "print(node_features('paypal-secure-login.tld')[-1])"
    ) % BACKEND_ROOT
    outs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1]
    assert float(outs[0]) == pytest.approx(node_features("paypal-secure-login.tld")[-1])


def test_node_features_sets_ipv4_flag_and_zero_tld():
    vec = dict(zip(GRAPH_FEATURE_NAMES, node_features("198.51.100.7")))
    assert vec["is_ipv4"] == 1.0
    assert vec["tld_length"] == 0.0
    assert vec["is_subnet"] == 0.0


def test_node_features_detects_punycode_and_brand_token():
    vec = dict(zip(GRAPH_FEATURE_NAMES, node_features("xn--80ak6aa92e.com")))
    assert vec["is_punycode"] == 1.0
    brand = dict(zip(GRAPH_FEATURE_NAMES, node_features("secure-login-paypal.tld")))
    assert brand["has_brand_token"] == 1.0


def test_node_features_handles_subnet_and_empty_input():
    sub = dict(zip(GRAPH_FEATURE_NAMES, node_features("198.51.100.0/24")))
    assert sub["is_subnet"] == 1.0
    assert sub["subnet_prefix_len"] == 24.0
    assert len(node_features("")) == N_FEATURES


def test_node_features_counts_digits_and_hyphens():
    vec = dict(zip(GRAPH_FEATURE_NAMES, node_features("pay-pal-secure-01.tld")))
    assert vec["hyphen_count"] == 3.0
    assert vec["digit_ratio"] > 0.0
    assert vec["max_digit_run"] == 2.0


def test_subnet_of_is_a_24():
    assert subnet_of("198.51.100.7") == "198.51.100.0/24"


# ---------------------------------------------------------------------------
# build_graph: node construction
# ---------------------------------------------------------------------------
def test_build_graph_collapses_subdomains_into_registrable_domain_nodes(tiny_graph):
    """``login.secure-bank-a.com`` and ``shop.secure-bank-b.com`` are 2 nodes."""
    assert "secure-bank-a.com" in tiny_graph.domain_to_index
    assert "login.secure-bank-a.com" not in tiny_graph.domain_to_index
    assert "unknown-hosting.xyz" in tiny_graph.domain_to_index
    assert "deep.sub.unknown-hosting.xyz" not in tiny_graph.domain_to_index


def test_build_graph_node_counts(tiny_graph):
    # 5 registrable domains + 2 IPv4 literals + their 2 /24 subnets.
    assert tiny_graph.num_domain_nodes == 5
    assert tiny_graph.num_ip_nodes == 2
    assert tiny_graph.num_subnet_nodes == 2
    assert tiny_graph.num_nodes == 9


def test_build_graph_node_blocks_are_contiguous(tiny_graph):
    """Domains, then IPs, then subnets - callers rely on slicing these blocks."""
    kinds = tiny_graph.node_kind
    n_d = tiny_graph.num_domain_nodes
    assert set(kinds[:n_d].tolist()) == {0}
    assert set(kinds[n_d : n_d + tiny_graph.num_ip_nodes].tolist()) == {1}
    for name, idx in tiny_graph.ip_to_index.items():
        assert tiny_graph.node_names[idx] == name


def test_build_graph_labels_only_come_from_urls(tiny_graph):
    assert tiny_graph.y[tiny_graph.domain_to_index["secure-bank-a.com"]] == 1
    assert tiny_graph.y[tiny_graph.domain_to_index["museum-site.org"]] == 0
    assert tiny_graph.y[tiny_graph.ip_to_index["198.51.100.7"]] == 1
    # Subnet nodes are inferred infrastructure, never labelled by a URL, and
    # the trainer excludes them from every metric via the labelled mask.
    for subnet, idx in tiny_graph.subnet_to_index.items():
        assert tiny_graph.y[idx] == -1, f"{subnet} must stay unlabelled"


def test_build_graph_assigns_split_from_first_url(tiny_graph):
    assert tiny_graph.node_split[tiny_graph.domain_to_index["secure-bank-a.com"]] == SPLIT_TRAIN
    assert tiny_graph.node_split[tiny_graph.domain_to_index["museum-site.org"]] == SPLIT_VAL
    assert tiny_graph.node_split[tiny_graph.domain_to_index["unknown-hosting.xyz"]] == SPLIT_VAL
    assert tiny_graph.node_split[tiny_graph.ip_to_index["198.51.100.7"]] == SPLIT_TEST


def test_build_graph_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_graph(["http://a.tld"], [1, 0])


def test_build_graph_survives_unparseable_urls():
    spec = build_graph(["not a url", "http://real-site.com/x"], [0, 1])
    assert "real-site.com" in spec.domain_to_index
    assert spec.num_nodes == 1


def test_build_graph_is_deterministic_across_builds():
    a = build_graph(TINY_URLS, TINY_LABELS, splits=TINY_SPLITS)
    b = build_graph(TINY_URLS, TINY_LABELS, splits=TINY_SPLITS)
    assert a.node_names == b.node_names
    assert a.named_edge_set() == b.named_edge_set()
    assert np.array_equal(a.x, b.x)


# ---------------------------------------------------------------------------
# build_graph: the no-label-leakage invariant
# ---------------------------------------------------------------------------
def test_edge_set_identical_under_label_permutation(tiny_graph):
    """Permuting labels must not change a single edge.

    This is the direct test of the module's central constraint. If an edge type
    were derived from label co-occurrence, the flipped-label graph would have a
    different edge set here and the test would fail.
    """
    flipped = [1 - y for y in TINY_LABELS]
    permuted = build_graph(TINY_URLS, flipped, splits=TINY_SPLITS)
    assert permuted.named_edge_set() == tiny_graph.named_edge_set()
    assert permuted.edge_index.shape == tiny_graph.edge_index.shape


def test_edge_set_identical_under_label_permutation_with_resolution(tiny_resolved_graph):
    """Same invariant on the DNS-answer path (IP + subnet + shared-IP edges)."""
    flipped = [1 - y for y in TINY_LABELS]
    permuted = build_graph(
        TINY_URLS, flipped,
        extra_domains={
            "secure-bank-a.com": ["198.51.100.7"],
            "secure-bank-b.com": ["198.51.100.7"],
            "museum-site.org": ["192.0.2.10"],
        },
        splits=TINY_SPLITS,
    )
    assert permuted.named_edge_set() == tiny_resolved_graph.named_edge_set()


def test_no_edge_type_is_a_label_relation():
    """Every declared edge type must be structural, and none may be label-only."""
    assert N_EDGE_TYPES == 7
    assert {EDGE_DOMAIN_IP, EDGE_SHARED_IP, EDGE_SHARED_SUBDOMAIN, EDGE_SHARED_TLD,
            EDGE_CONTAINMENT, EDGE_DOMAIN_SUBNET, EDGE_IP_SUBNET} == set(range(N_EDGE_TYPES))


# ---------------------------------------------------------------------------
# build_graph: edge construction
# ---------------------------------------------------------------------------
def test_no_ip_edges_without_resolution_data(tiny_graph):
    """The headline limitation, asserted: URL strings alone give no IP edges.

    The two IP-literal URLs still produce IP nodes and ip->subnet edges, but no
    ``domain_ip`` edge exists because no domain is ever *resolved* to an address.
    """
    counts = tiny_graph.edge_type_counts()
    assert counts["domain_ip"] == 0
    assert counts["domain_subnet"] == 0
    assert counts["shared_ip"] == 0
    assert counts["ip_subnet"] > 0


def test_resolution_data_creates_ip_and_subnet_nodes(tiny_resolved_graph):
    assert "192.0.2.10" in tiny_resolved_graph.ip_to_index
    assert "198.51.100.0/24" in tiny_resolved_graph.subnet_to_index
    # 198.51.100.7 and 203.0.113.9 come from the IP-literal URLs; 192.0.2.10
    # only exists because resolution data was supplied.
    assert tiny_resolved_graph.num_ip_nodes == 3
    assert tiny_resolved_graph.num_subnet_nodes == 3
    counts = tiny_resolved_graph.edge_type_counts()
    assert counts["domain_ip"] > 0
    assert counts["domain_subnet"] > 0


def test_shared_infrastructure_links_two_domains_directly(tiny_resolved_graph):
    """Two domains on one IP get a direct edge, not just a 2-hop path."""
    pairs = tiny_resolved_graph.named_edge_set()
    a = "secure-bank-a.com"
    b = "secure-bank-b.com"
    assert (a, b, EDGE_SHARED_IP) in pairs
    assert (b, a, EDGE_SHARED_IP) in pairs, "relations are undirected"


def test_shared_subdomain_token_creates_an_edge():
    spec = build_graph(
        ["http://login.alpha-shop.com/a", "http://login.beta-shop.org/b"],
        [1, 0],
    )
    pairs = spec.named_edge_set()
    assert ("alpha-shop.com", "beta-shop.org", EDGE_SHARED_SUBDOMAIN) in pairs


def test_no_shared_subdomain_edge_when_tokens_differ():
    spec = build_graph(
        ["http://login.alpha-shop.com/a", "http://checkout.beta-shop.org/b"],
        [1, 0],
    )
    assert spec.edge_type_counts()["shared_subdomain"] == 0


def test_shared_tld_edges_exist_and_are_degree_capped():
    urls = [f"http://site{i}.host{i}.com/p" for i in range(20)]
    spec = build_graph(urls, [i % 2 for i in range(20)], max_tld_degree=4)
    counts = spec.edge_type_counts()["shared_tld"]
    # Undirected, degree-capped at 4: at most 20 * 4 / 2 pairs, and a 20-clique
    # (190 edges) is explicitly out of reach.
    assert 0 < counts <= 20 * 4
    for i in range(20):
        domain = f"host{i}.com"
        deg = sum(1 for s, d, t in spec.named_edge_set()
                  if s == domain and d != domain and t == EDGE_SHARED_TLD)
        assert deg <= 4


def test_every_edge_is_symmetric_and_never_a_self_loop(tiny_graph):
    pairs = tiny_graph.named_edge_set()
    for s, d, t in pairs:
        assert s != d, "self-loops carry no information and bias the degree norm"
        assert (d, s, t) in pairs


def test_edge_endpoints_are_valid_node_indices(tiny_resolved_graph):
    assert tiny_resolved_graph.edge_index.min() >= 0
    assert tiny_resolved_graph.edge_index.max() < tiny_resolved_graph.num_nodes


def test_edge_type_counts_sum_to_edge_count(tiny_resolved_graph):
    assert sum(tiny_resolved_graph.edge_type_counts().values()) == tiny_resolved_graph.num_edges


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
def test_save_load_round_trip(tmp_path, tiny_resolved_graph):
    path = tmp_path / "graph.npz"
    save_graph(tiny_resolved_graph, path, extra_meta={"note": "unit-test"})
    back = load_graph(path)

    assert back.node_names == tiny_resolved_graph.node_names
    assert back.domain_to_index == tiny_resolved_graph.domain_to_index
    assert back.ip_to_index == tiny_resolved_graph.ip_to_index
    assert back.subnet_to_index == tiny_resolved_graph.subnet_to_index
    assert np.array_equal(back.edge_index, tiny_resolved_graph.edge_index)
    assert np.array_equal(back.edge_type, tiny_resolved_graph.edge_type)
    assert np.array_equal(back.y, tiny_resolved_graph.y)
    assert np.array_equal(back.node_split, tiny_resolved_graph.node_split)
    assert np.array_equal(back.x, tiny_resolved_graph.x)
    assert back.named_edge_set() == tiny_resolved_graph.named_edge_set()


def test_load_graph_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_graph(tmp_path / "nope.npz")


def test_load_graph_rejects_unknown_schema_version(tmp_path, tiny_graph):
    path = tmp_path / "graph.npz"
    save_graph(tiny_graph, path)
    with np.load(path, allow_pickle=True) as blob:
        arrays = {k: blob[k] for k in blob.files}
    import json as _json
    meta = _json.loads(str(arrays["meta_json"].item()))
    meta["schema_version"] = 99
    arrays["meta_json"] = np.array(_json.dumps(meta), dtype=object)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        load_graph(path)


def test_split_grouping_verification_flags_a_violation(tmp_path, tiny_graph):
    """The leakage guard is only useful if it actually reports violations."""
    csv = tmp_path / "domain_split.csv"
    csv.write_text("registered_domain,split\nsecure-bank-a.com,train\nmuseum-site.org,test\n",
                   encoding="utf-8")
    result = verify_split_grouping(tiny_graph, csv)
    assert result["checked"] is True
    # museum-site.org was built as a val node but the table says test.
    assert result["n_violations"] == 1
    assert result["violations"][0]["domain"] == "museum-site.org"


# ---------------------------------------------------------------------------
# GraphConv
# ---------------------------------------------------------------------------
def test_graph_conv_output_shape():
    conv = GraphConv(5, 8)
    x = torch.randn(7, 5)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    assert conv(edge_index, x, None).shape == (7, 8)


def test_graph_conv_with_no_edges_falls_back_to_self_transform():
    conv = GraphConv(4, 6)
    out = conv(torch.zeros((2, 0), dtype=torch.long), torch.randn(3, 4), None)
    assert out.shape == (3, 6)


def test_graph_conv_is_permutation_invariant_over_node_ordering():
    """Relabelling nodes permutes the output rows and changes nothing else.

    A GCN that were order-sensitive (e.g. one that accidentally used node
    position as a feature) would fail here, and the reported metrics would
    depend on an arbitrary sort.
    """
    torch.manual_seed(0)
    conv = GraphConv(6, 6).eval()
    n = 8
    src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 2], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 7, 5], dtype=torch.long)
    edge_index = torch.stack([src, dst])
    x = torch.randn(n, 6)

    perm = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4], dtype=torch.long)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n, dtype=torch.long)

    base = conv(edge_index, x, None)
    moved = conv(inverse[edge_index], x[perm], None)

    # ``moved[j]`` scores original node ``perm[j]``, so ``moved == base[perm]`` and
    # equivalently ``base == moved[inverse]``.
    assert torch.allclose(moved, base[perm], atol=1e-5)
    assert torch.allclose(base, moved[inverse], atol=1e-5)


def test_graph_conv_uses_per_relation_mean_normalisation():
    """One relation's volume must not swamp another's contribution."""
    torch.manual_seed(0)
    conv = GraphConv(3, 3, n_edge_types=2).eval()
    x = torch.randn(4, 3)
    # Node 0 receives 10 type-0 neighbours and 1 type-1 neighbour.
    src = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=torch.long)
    dst = torch.zeros(11, dtype=torch.long)
    etype = torch.tensor([0] * 10 + [1], dtype=torch.long)
    out = conv(torch.stack([src, dst]), x, etype)

    with torch.no_grad():
        manual = torch.softmax(conv.rel_logits, dim=0)
        nbr = conv.neighbour_lin(
            manual[0] * x[1:2]  # type-0 mean over one distinct value
            + manual[1] * x[2:3]
        )
    # With one distinct value per relation the per-relation mean equals the value.
    assert torch.allclose(out[0], conv.self_lin(x[0]) + nbr.squeeze(0), atol=1e-5)


def test_graph_conv_rejects_bad_edge_index():
    conv = GraphConv(3, 3)
    with pytest.raises(ValueError):
        conv(torch.zeros((3, 4), dtype=torch.long), torch.randn(2, 3), None)


# ---------------------------------------------------------------------------
# PhishingGCN
# ---------------------------------------------------------------------------
def test_phishing_gcn_forward_shapes(tiny_graph):
    model = PhishingGCN(in_dim=tiny_graph.feature_dim, hidden_dim=8, n_layers=3,
                        n_edge_types=N_EDGE_TYPES).eval()
    edge_index = torch.as_tensor(tiny_graph.edge_index, dtype=torch.long)
    edge_type = torch.as_tensor(tiny_graph.edge_type, dtype=torch.long)
    x = torch.as_tensor(tiny_graph.x, dtype=torch.float32)
    out = model(edge_index, x, edge_type)

    assert isinstance(out, GCNOutput)
    assert out.logit.shape == (tiny_graph.num_nodes,)
    assert out.embedding.shape == (tiny_graph.num_nodes, 8)
    assert bool(((out.probability >= 0) & (out.probability <= 1)).all())


def test_phishing_gcn_accepts_a_node_mask(tiny_graph):
    model = PhishingGCN(in_dim=tiny_graph.feature_dim, hidden_dim=8,
                        n_edge_types=N_EDGE_TYPES).eval()
    edge_index = torch.as_tensor(tiny_graph.edge_index, dtype=torch.long)
    edge_type = torch.as_tensor(tiny_graph.edge_type, dtype=torch.long)
    x = torch.as_tensor(tiny_graph.x, dtype=torch.float32)
    mask = torch.tensor([0, 2, 4], dtype=torch.long)

    full = model(edge_index, x, edge_type)
    masked = model(edge_index, x, edge_type, node_mask=mask)
    assert masked.logit.shape == (3,)
    assert torch.allclose(masked.logit, full.logit[mask], atol=1e-6)


def test_phishing_gcn_gradients_reach_every_parameter(tiny_graph):
    model = PhishingGCN(in_dim=tiny_graph.feature_dim, hidden_dim=8, n_edge_types=N_EDGE_TYPES)
    edge_index = torch.as_tensor(tiny_graph.edge_index, dtype=torch.long)
    edge_type = torch.as_tensor(tiny_graph.edge_type, dtype=torch.long)
    x = torch.as_tensor(tiny_graph.x, dtype=torch.float32)
    out = model(edge_index, x, edge_type)
    out.logit.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert bool(torch.isfinite(p.grad).all()), f"{name} produced a non-finite gradient"


def test_phishing_gcn_rejects_zero_layers(tiny_graph):
    with pytest.raises(ValueError):
        PhishingGCN(in_dim=4, n_layers=0)


def test_phishing_gcn_config_round_trips_through_construction(tiny_graph):
    model = PhishingGCN(in_dim=tiny_graph.feature_dim, hidden_dim=8, n_edge_types=N_EDGE_TYPES)
    cfg = model.config()
    clone = PhishingGCN(**cfg)
    clone.load_state_dict(model.state_dict())
    edge_index = torch.as_tensor(tiny_graph.edge_index, dtype=torch.long)
    edge_type = torch.as_tensor(tiny_graph.edge_type, dtype=torch.long)
    x = torch.as_tensor(tiny_graph.x, dtype=torch.float32)
    model.eval()
    clone.eval()
    assert torch.allclose(
        model(edge_index, x, edge_type).logit,
        clone(edge_index, x, edge_type).logit,
        atol=1e-6,
    )


def test_phishing_gcn_inference_view_matches_forward(tiny_graph):
    model = PhishingGCN(in_dim=tiny_graph.feature_dim, hidden_dim=8, n_edge_types=N_EDGE_TYPES).eval()
    edge_index = torch.as_tensor(tiny_graph.edge_index, dtype=torch.long)
    edge_type = torch.as_tensor(tiny_graph.edge_type, dtype=torch.long)
    x = torch.as_tensor(tiny_graph.x, dtype=torch.float32)
    view = model.inference_view(edge_index, x, edge_type)
    assert view.shape == (tiny_graph.num_nodes,)
    assert torch.allclose(view, model(edge_index, x, edge_type).probability, atol=1e-6)
