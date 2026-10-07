"""Prototype layer: classification by similarity to learned attack/benign patterns.

Each prototype is a point in embedding space owned by one class. After training,
`project` snaps every prototype onto its nearest real training delegation, so each
prototype *is* a concrete on-chain pattern an analyst can inspect. The distance to
the nearest prototype doubles as an open-set score: a delegation unlike every known
pattern is flagged as novel instead of being forced into a class.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLayer(nn.Module):
    def __init__(self, dim: int, per_class: int = 10, n_classes: int = 2):
        super().__init__()
        self.protos = nn.Parameter(torch.randn(per_class * n_classes, dim) * 0.1)
        self.register_buffer("proto_class", torch.arange(n_classes).repeat_interleave(per_class))
        w = torch.full((n_classes, per_class * n_classes), -0.5)
        w[self.proto_class, torch.arange(per_class * n_classes)] = 1.0
        self.last = nn.Parameter(w)                   # ProtoPNet-style class connections
        self.register_buffer("source", torch.full((per_class * n_classes,), -1, dtype=torch.long))

    def forward(self, z: torch.Tensor):
        d2 = torch.cdist(z, self.protos).pow(2)       # [B, P]
        sim = torch.log((d2 + 1) / (d2 + 1e-4))
        logits = sim @ self.last.t()                  # [B, C]
        return logits[:, 1] - logits[:, 0], sim, d2

    def losses(self, d2: torch.Tensor, y: torch.Tensor, margin: float = 1.0):
        own = self.proto_class[None, :] == y[:, None].long()
        big = d2.max().detach() + 1
        cluster = torch.where(own, d2, big).min(1).values.mean()
        sep = -torch.where(~own, d2, big).min(1).values.clamp(max=10).mean()
        p = F.normalize(self.protos, dim=1)
        same = self.proto_class[:, None] == self.proto_class[None, :]
        cos = (p @ p.t()).masked_fill(~same | torch.eye(len(p), dtype=torch.bool, device=p.device), -1)
        diversity = F.relu(cos - 0.5).sum() / len(p)
        return cluster, sep, diversity

    @torch.no_grad()
    def project(self, z: torch.Tensor, y: torch.Tensor, ids: torch.Tensor):
        """Replace each prototype by the closest training embedding of its own class."""
        for p in range(len(self.protos)):
            cand = (y.long() == self.proto_class[p]).nonzero().flatten()
            if not len(cand):
                continue
            d = (z[cand] - self.protos[p]).pow(2).sum(1)
            j = cand[d.argmin()]
            self.protos[p] = z[j]
            self.source[p] = ids[j]

    @torch.no_grad()
    def novelty(self, d2: torch.Tensor) -> torch.Tensor:
        return d2.min(1).values
