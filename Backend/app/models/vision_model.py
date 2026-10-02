"""Rendered-screenshot (vision) modality model.

A pretrained CNN backbone (``resnet18`` on the dev profile, ``deit_tiny`` on the
full one) with the classification head replaced, producing one embedding for the
fusion network.

``pretrained=True`` downloads ImageNet weights. When the download is unavailable
the backbone falls back to random initialisation **and says so**, because a
randomly initialised backbone presented as a pretrained one would be a
misleading claim about what the model learned.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["VisionModalityModel", "VisionModelOutput", "build_backbone"]


class VisionModelOutput:
    __slots__ = ("embedding", "logit")

    def __init__(self, embedding: torch.Tensor, logit: torch.Tensor) -> None:
        self.embedding = embedding
        self.logit = logit


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int, bool]:
    """Return ``(feature_extractor, feature_dim, pretrained_actually_used)``."""
    import torchvision.models as tvm

    name = name.lower()
    weights = "DEFAULT" if pretrained else None
    try:
        if name == "resnet18":
            net = tvm.resnet18(weights=weights)
            dim = net.fc.in_features
            net.fc = nn.Identity()
        elif name == "resnet50":
            net = tvm.resnet50(weights=weights)
            dim = net.fc.in_features
            net.fc = nn.Identity()
        elif name in {"deit_tiny", "deit_tiny_patch16_224"}:
            net = tvm.deit_tiny_patch16_224(weights=weights)
            dim = net.heads.in_features
            net.heads = nn.Identity()
        else:
            raise ValueError(f"unsupported backbone {name!r}")
    except Exception:
        # Weight download failed or is disabled. Rebuild untrained and flag it.
        weights = None
        if name == "resnet18":
            net = tvm.resnet18(weights=None)
            dim = net.fc.in_features
            net.fc = nn.Identity()
        elif name == "resnet50":
            net = tvm.resnet50(weights=None)
            dim = net.fc.in_features
            net.fc = nn.Identity()
        elif name in {"deit_tiny", "deit_tiny_patch16_224"}:
            net = tvm.deit_tiny_patch16_224(weights=None)
            dim = net.heads.in_features
            net.heads = nn.Identity()
        else:
            raise
    return net, int(dim), bool(pretrained and weights is not None)


class VisionModalityModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_out: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        net, dim, used = build_backbone(backbone, pretrained)
        self.features = net
        self.pretrained_loaded = used
        self.config = dict(
            backbone=backbone, pretrained=pretrained, embedding_out=embedding_out,
            dropout=dropout, feature_dim=dim,
        )
        self.proj = nn.Sequential(
            nn.Linear(dim, embedding_out),
            nn.LayerNorm(embedding_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(embedding_out, 1)

    def forward(self, image: torch.Tensor) -> VisionModelOutput:
        feats = self.features(image)
        emb = self.proj(feats)
        return VisionModelOutput(embedding=emb, logit=self.head(emb).squeeze(-1))

    def config_dict(self) -> dict:
        return dict(self.config)
