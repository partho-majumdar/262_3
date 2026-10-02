"""Graph neural network over the domain-IP graph.

    x (N, F) --[GraphConv]--> h1 --ReLU--> --[GraphConv]--> h2 --ReLU--> --[GraphConv]--> z
                                                                                        |
                                                                        Linear -> node logit

Written from scratch in pure PyTorch on purpose. The host is CPU-only with
``torch_geometric`` absent, and the graph is small (thousands of nodes, tens of
thousands of edges), so a scatter-based message passing layer is both fast
enough and far easier to audit than a dependency we cannot install. Nothing here
requires a sparse-CSR backend: aggregation is ``index_add_`` on a dense
``(N, D)`` buffer.

Why a GCN and not attention
---------------------------
The hypothesis the project wants to test is "a domain that shares
infrastructure with known-phishing domains is suspicious". That is a
neighbourhood-smoothing claim, not a "which neighbour matters" claim, and the
neighbourhood sizes here are small and uneven. A degree-normalised mean
aggregator is the right inductive bias for that; attention would add parameters
the graph cannot pay for and would invite the model to memorise individual
neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["GraphConv", "PhishingGCN", "GCNOutput"]


# ---------------------------------------------------------------------------
# Scatter helpers (the "message passing" primitive)
# ---------------------------------------------------------------------------
def _segment_sum(values: Tensor, index: Tensor, n_segments: int) -> Tensor:
    """Sum ``values`` rows into ``n_segments`` buckets given by ``index``.

    ``index_add_`` is deterministic on CPU for a fixed input order and avoids
    materialising the ``|E| x |N|`` dense adjacency a matmul would need - at
    4000 nodes the dense form is 64 MB per layer per step, which is not a
    CPU-friendly thing to allocate inside an autograd loop.
    """
    out = torch.zeros((n_segments, values.size(-1)), dtype=values.dtype, device=values.device)
    return out.index_add_(0, index, values)


def _segment_count(index: Tensor, n_segments: int, like: Tensor) -> Tensor:
    """Bucket sizes for ``index``, shaped like a ``(n_segments, 1)`` feature map."""
    ones = torch.ones((index.size(0), 1), dtype=like.dtype, device=like.device)
    return _segment_sum(ones, index, n_segments)


def _segment_mean(values: Tensor, index: Tensor, n_segments: int) -> Tensor:
    """Degree-normalised mean aggregation: ``sum / max(count, 1)``.

    ``max(count, 1)`` rather than ``count`` is what keeps an isolated node
    well-defined: it receives an all-zero message instead of a division by zero,
    and its own self-transform carries its representation forward.
    """
    total = _segment_sum(values, index, n_segments)
    count = _segment_count(index, n_segments, values).clamp(min=1.0)
    return total / count


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------
class GraphConv(nn.Module):
    """One degree-normalised message-passing layer.

    For a single relation type the layer computes

        h_v' = act( W_self h_v + W_nbr * a_v ),
        a_v  = (1 / max(|N(v)|, 1)) * sum_{u in N(v)} h_u

    i.e. the neighbour term is the **mean** of the incoming messages, not the
    sum, so a domain that resolves through a busy IP is not louder than one that
    resolves through a quiet IP. Without this normalisation the magnitude of a
    node's representation would scale with its degree and two hops would already
    saturate a ReLU network.

    With ``n_edge_types > 1`` the layer aggregates **per relation** and mixes
    the relation means with learned weights:

        a_v^(t) = (1 / max(|N_t(v)|, 1)) * sum_{u in N_t(v)} h_u
        beta     = softmax(theta)                       # (n_edge_types,)
        a_v      = sum_t beta_t a_v^(t)

    Normalising within each relation *before* mixing is deliberate: a common TLD
    relation would otherwise dominate the shared-subdomain relation purely by
    volume, and the model would learn "TLD" instead of "infrastructure".

    Args:
        in_dim: input feature width.
        out_dim: output feature width.
        n_edge_types: number of relation ids in ``edge_type``.
        bias: keep a bias on the output projection.
    """

    def __init__(self, in_dim: int, out_dim: int, n_edge_types: int = 1, bias: bool = True) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim/out_dim must be positive, got {in_dim}/{out_dim}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_edge_types = max(1, int(n_edge_types))

        self.self_lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.neighbour_lin = nn.Linear(in_dim, out_dim, bias=False)
        # Initialised to zeros so softmax starts at a uniform mix: at step 0 the
        # layer is exactly a relation-agnostic mean aggregator, and the model
        # has to *earn* any preference for a relation from validation loss.
        self.rel_logits = nn.Parameter(torch.zeros(self.n_edge_types))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Glorot init on both projections (the mix weights stay at uniform)."""
        for lin in (self.self_lin, self.neighbour_lin):
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        with torch.no_grad():
            self.rel_logits.zero_()

    def forward(
        self,
        edge_index: Tensor,
        x: Tensor,
        edge_type: Tensor | None = None,
    ) -> Tensor:
        """Propagate over ``edge_index``.

        Args:
            edge_index: ``(2, E)`` int64, row 0 = source, row 1 = target. Messages
                flow ``source -> target``.
            x: ``(N, in_dim)`` node features.
            edge_type: ``(E,)`` int64 relation ids, required when
                ``n_edge_types > 1``.

        Returns:
            ``(N, out_dim)`` post-transform representations.
        """
        if x.dim() != 2:
            raise ValueError(f"x must be (N, F), got {tuple(x.shape)}")
        n_nodes = x.size(0)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must be (2, E), got {tuple(edge_index.shape)}")
        src, dst = edge_index[0].long(), edge_index[1].long()
        if src.numel() == 0:
            # No edges: the layer degenerates to the self transform. Returning
            # it (rather than raising) keeps single-node graphs trainable.
            return self.self_lin(x)

        messages = x.index_select(0, src)

        if edge_type is None or self.n_edge_types == 1:
            aggregated = _segment_mean(messages, dst, n_nodes)
        else:
            if edge_type.dim() != 1 or edge_type.numel() != src.numel():
                raise ValueError(
                    f"edge_type must be ({src.numel()},), got {tuple(edge_type.shape)}"
                )
            et = edge_type.long()
            if int(et.max()) >= self.n_edge_types or int(et.min()) < 0:
                raise ValueError(
                    f"edge_type values must lie in [0, {self.n_edge_types})"
                )
            weight = torch.softmax(self.rel_logits, dim=0)
            aggregated = torch.zeros_like(x)
            for t in range(self.n_edge_types):
                sel = et == t
                if not bool(sel.any()):
                    continue
                # Per-relation mean *before* weighting: see the class docstring.
                aggregated = aggregated + weight[t] * _segment_mean(
                    messages[sel], dst[sel], n_nodes
                )

        return self.self_lin(x) + self.neighbour_lin(aggregated)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, n_edge_types={self.n_edge_types}"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class GCNOutput:
    """Per-node model output.

    Mirrors :class:`app.models.url_model.URLModelOutput` so callers can treat the
    two branches the same way; the difference is the leading dimension. Here
    ``N`` is nodes, and ``logit`` is phishing-ness *of a domain*, not of a URL.
    """

    logit: Tensor  # (N,) or (M,) when node_mask is used
    probability: Tensor  # same shape, in [0, 1]
    embedding: Tensor  # (N, hidden) or (M, hidden)


class PhishingGCN(nn.Module):
    """Stack of :class:`GraphConv` layers with a per-node phishing head.

    Why this depth: 2-3 hops is the useful range here. A domain's immediate
    TLD-mates are weakly related; the useful evidence ("this whole subnet of
    newly-registered lookalike domains is phishing") lives at 2-3 hops. Beyond
    that the receptive field is the entire ``.com`` neighbourhood and every node
    converges to the same vector - the classic over-smoothing collapse.

    Args:
        in_dim: node feature width (``len(GRAPH_FEATURE_NAMES)``).
        hidden_dim: width of the message-passing layers.
        n_layers: number of ``GraphConv`` blocks (>= 1).
        n_edge_types: relation count, must cover ``edge_type`` ids.
        dropout: dropout on hidden representations and on the input features.
        head_dropout: separate dropout inside the classifier head.
        input_dropout: applied to the raw features before layer 1; the
            handcrafted features are correlated (length, digit ratio, entropy),
            and a little noise there regularises more usefully than a deeper net.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_edge_types: int = 1,
        dropout: float = 0.5,
        head_dropout: float = 0.3,
        input_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_edge_types = max(1, int(n_edge_types))

        dims = [self.in_dim] + [self.hidden_dim] * self.n_layers
        self.convs = nn.ModuleList(
            [GraphConv(dims[i], dims[i + 1], n_edge_types=self.n_edge_types) for i in range(self.n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layers - 1)])

        self.input_dropout = nn.Dropout(input_dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        edge_index: Tensor,
        x: Tensor,
        edge_type: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> GCNOutput:
        """Score every node.

        Args:
            edge_index: ``(2, E)`` int64, ``edge_index[0] -> edge_index[1]``.
            x: ``(N, in_dim)`` node features.
            edge_type: ``(E,)`` int64 relation ids, or ``None`` to aggregate all
                relations together.
            node_mask: optional ``(M,)`` bool / int index tensor. Message passing
                still runs over the **whole** graph and only the returned rows
                are gathered, which is what makes this safe for the transductive
                node split used by the trainer: masking selects which nodes are
                scored, it never edits the neighbourhood they aggregate over.

                Be clear about what this is *not*: it is not neighbourhood
                sampling, so a masked node still sees the whole graph. It is for
                cheap batched inference (score many domains in one forward),
                not for hiding unlabelled nodes from the model.

        Returns:
            :class:`GCNOutput` over all nodes, or over ``node_mask`` rows when a
            mask is supplied.
        """
        h = self.input_dropout(x)
        for i, conv in enumerate(self.convs):
            h = conv(edge_index, h, edge_type)
            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)

        logit = self.head(h).squeeze(-1)
        out = GCNOutput(logit=logit, probability=torch.sigmoid(logit), embedding=h)
        if node_mask is not None:
            idx = node_mask.long()
            out = GCNOutput(
                logit=out.logit.index_select(0, idx),
                probability=out.probability.index_select(0, idx),
                embedding=out.embedding.index_select(0, idx),
            )
        return out

    # ------------------------------------------------------------------
    def inference_view(self, edge_index: Tensor, x: Tensor, edge_type: Tensor | None = None) -> Tensor:
        """Phishing probability per node, in eval-friendly form.

        Provided so an integrating service does not have to know that ``forward``
        returns a dataclass, while still going through the same code path the
        trainer validated.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            probs = self.forward(edge_index, x, edge_type).probability
        if was_training:
            self.train()
        return probs

    def config(self) -> dict:
        """Architecture config stored in the checkpoint."""
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_edge_types": self.n_edge_types,
            "dropout": float(self.dropout.p),
            "head_dropout": float(self.head[2].p) if len(self.head) > 2 else 0.0,
            "input_dropout": float(self.input_dropout.p),
        }
