"""Graph baselines: TGAT-style temporal attention and GraphSAGE mean aggregation over the
delegation graph. The flags let one class express both, so they share the same sampler,
features and training loop. (The paper's DelegNet is code-based: baselines.OpcodeKNN.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .graph import AUTH_FEATS, EDGE_FEATS, N_REL, NODE_FEATS


@dataclass
class Flags:
    use_graph: bool = True
    use_time: bool = True
    use_rel: bool = True
    attention: bool = True


VARIANTS = {
    "TGAT":      Flags(use_rel=False),
    "GraphSAGE": Flags(use_rel=False, use_time=False, attention=False),
}


def segment_softmax(s: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    mx = torch.full((n, s.shape[1]), -1e30, device=s.device, dtype=s.dtype)
    mx = mx.scatter_reduce(0, idx[:, None].expand_as(s), s, "amax", include_self=True)
    e = (s - mx[idx]).exp()
    den = torch.zeros(n, s.shape[1], device=s.device, dtype=s.dtype).index_add_(0, idx, e)
    return e / (den[idx] + 1e-16)


class TimeEncoding(nn.Module):
    """TGAT-style functional time encoding; frequencies span seconds to months."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(1.0 / 10 ** torch.linspace(0, 7, dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, dt):
        return torch.cos(dt[:, None] * self.w + self.b)


class HeteroTemporalAttention(nn.Module):
    def __init__(self, d, heads, n_rel, edim, tdim, dropout, flags: Flags):
        super().__init__()
        self.h, self.dh, self.flags = heads, d // heads, flags
        din = d + edim + (tdim if flags.use_time else 0)
        n_rel = n_rel if flags.use_rel else 1
        self.n_rel, self.d = n_rel, d
        self.q = nn.Linear(d, d)
        # shared key/value projection, modulated per relation (FiLM): cheap on GPU, still relation-aware
        self.kv = nn.Linear(din, 2 * d)
        self.rel_scale = nn.Embedding(n_rel, 2 * d)
        self.rel_shift = nn.Embedding(n_rel, 2 * d)
        nn.init.zeros_(self.rel_scale.weight)
        nn.init.zeros_(self.rel_shift.weight)
        self.o = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, src, dst, rel, tfeat, efeat):
        n, E = h.shape[0], src.shape[0]
        parts = [h[src], efeat] + ([tfeat] if self.flags.use_time else [])
        inp = torch.cat(parts, 1)
        rel = rel if self.flags.use_rel else torch.zeros_like(rel)
        kv = self.kv(inp) * (1 + self.rel_scale(rel)) + self.rel_shift(rel)
        k, v = kv[:, :self.d].reshape(E, self.h, self.dh), kv[:, self.d:].reshape(E, self.h, self.dh)
        if self.flags.attention:
            q = self.q(h).view(n, self.h, self.dh)[dst]
            score = (q * k).sum(-1) / math.sqrt(self.dh)
            alpha = segment_softmax(score, dst, n)
        else:  # mean aggregation (GraphSAGE)
            deg = torch.zeros(n, 1, device=h.device).index_add_(0, dst, torch.ones(E, 1, device=h.device))
            alpha = (1 / deg[dst].clamp_min(1e-6)).expand(E, self.h)
        agg = torch.zeros(n, self.h, self.dh, device=h.device).index_add_(0, dst, alpha[..., None] * v)
        h = self.n1(h + self.drop(self.o(agg.view(n, -1))))
        return self.n2(h + self.drop(self.ff(h)))


class TemporalGNN(nn.Module):
    def __init__(self, m: dict, flags: Flags):
        super().__init__()
        d, self.flags = m["dim"], flags
        self.node_in = nn.Linear(len(NODE_FEATS), d)
        self.role = nn.Embedding(4, d)
        self.kind = nn.Embedding(2, d)
        self.tenc = TimeEncoding(32)
        if flags.use_graph:
            self.layers = nn.ModuleList(HeteroTemporalAttention(d, m["heads"], 2 * N_REL, len(EDGE_FEATS), 32,
                                                                m["dropout"], flags) for _ in range(m["layers"]))
        head_in = 3 * d + len(AUTH_FEATS)
        self.readout = nn.Sequential(nn.Linear(head_in, d), nn.GELU(), nn.Dropout(m["dropout"]), nn.Linear(d, d))
        self.out = nn.Sequential(nn.GELU(), nn.Linear(d, 1))

    def forward(self, b: dict) -> dict:
        h = self.node_in(b["x"]) + self.role(b["role"]) + self.kind(b["is_contract"])
        if self.flags.use_graph:
            src = torch.cat([b["esrc"], b["edst"]])
            dst = torch.cat([b["edst"], b["esrc"]])
            rel = torch.cat([b["etype"], b["etype"] + N_REL])
            tf = self.tenc(torch.cat([b["edt"], b["edt"]]))
            ef = torch.cat([b["efeat"], b["efeat"]])
            for layer in self.layers:
                h = layer(h, src, dst, rel, tf, ef)
        c = b["centres"]
        z = self.readout(torch.cat([h[c[:, 0]], h[c[:, 1]], h[c[:, 2]], b["authf"]], 1))
        return dict(z=z, logit=self.out(z).squeeze(1))


DelegNet = TemporalGNN   # the class name under which the graph baselines in runs/ were saved
