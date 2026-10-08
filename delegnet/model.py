"""DelegNet: heterogeneous temporal graph attention + bytecode encoder + Pattern Recognition Module.

The flags let one class express the full model, its ablations, and the graph baselines
(static GAT / GraphSAGE, TGAT-style temporal attention without code or patterns), so
every comparison shares the same sampler, features and training loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import AUTH_FEATS, EDGE_FEATS, N_REL, NODE_FEATS, VOCAB
from .patterns.code_idioms import IDIOM_NAMES
from .patterns.motifs import MOTIF_NAMES
from .patterns.prototypes import PrototypeLayer


@dataclass
class Flags:
    use_graph: bool = True
    use_time: bool = True
    use_rel: bool = True
    attention: bool = True
    code: str = "transformer"     # transformer | idioms | none
    use_motifs: bool = True
    use_proto: bool = True


VARIANTS = {
    "DelegNet":            Flags(),
    "DelegNet-noProto":    Flags(use_proto=False),
    "DelegNet-noMotif":    Flags(use_motifs=False),
    "DelegNet-noCode":     Flags(code="none"),
    "DelegNet-idiomsOnly": Flags(code="idioms"),
    "DelegNet-noTime":     Flags(use_time=False),
    "DelegNet-noRel":      Flags(use_rel=False),
    "TGAT":                Flags(code="none", use_motifs=False, use_proto=False, use_rel=False),
    "StaticGAT":           Flags(code="none", use_motifs=False, use_proto=False, use_rel=False, use_time=False),
    "GraphSAGE":           Flags(code="none", use_motifs=False, use_proto=False, use_rel=False, use_time=False,
                                 attention=False),
    "DelegNet-noPretrain": Flags(),
    "MLP+Code":            Flags(use_graph=False, use_motifs=False, use_proto=False),
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

    def forward(self, h, src, dst, rel, tfeat, efeat, edge_weight=None):
        n, E = h.shape[0], src.shape[0]
        parts = [h[src], efeat] + ([tfeat] if self.flags.use_time else [])
        inp = torch.cat(parts, 1)
        rel = rel if self.flags.use_rel else torch.zeros_like(rel)
        kv = self.kv(inp) * (1 + self.rel_scale(rel)) + self.rel_shift(rel)
        k, v = kv[:, :self.d].reshape(E, self.h, self.dh), kv[:, self.d:].reshape(E, self.h, self.dh)
        if self.flags.attention:
            q = self.q(h).view(n, self.h, self.dh)[dst]
            score = (q * k).sum(-1) / math.sqrt(self.dh)
            if edge_weight is not None:
                score = score + torch.log(edge_weight.clamp_min(1e-6))[:, None]
            alpha = segment_softmax(score, dst, n)
        else:  # mean aggregation (GraphSAGE)
            w = torch.ones(E, 1, device=h.device) if edge_weight is None else edge_weight[:, None]
            deg = torch.zeros(n, 1, device=h.device).index_add_(0, dst, w)
            alpha = (w / deg[dst].clamp_min(1e-6)).expand(E, self.h)
        agg = torch.zeros(n, self.h, self.dh, device=h.device).index_add_(0, dst, alpha[..., None] * v)
        h = self.n1(h + self.drop(self.o(agg.view(n, -1))))
        return self.n2(h + self.drop(self.ff(h)))


class CodeEncoder(nn.Module):
    def __init__(self, d, layers, heads, max_len, dropout):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens):
        pad = tokens == 0
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.enc(self.tok(tokens) + self.pos(pos)[None], src_key_padding_mask=pad)
        x = x.masked_fill(pad[..., None], 0).sum(1) / (~pad).sum(1, keepdim=True).clamp_min(1)
        return self.norm(x)


class DelegNet(nn.Module):
    def __init__(self, m: dict, flags: Flags):
        super().__init__()
        d, self.flags = m["dim"], flags
        self.node_in = nn.Linear(len(NODE_FEATS), d)
        self.role = nn.Embedding(4, d)
        self.kind = nn.Embedding(2, d)
        self.tenc = TimeEncoding(32)
        cd = m.get("code_dim", d)   # width of the (pre-trained, frozen) code encoder, independent of `dim`
        if flags.code == "transformer":
            self.code_enc = CodeEncoder(cd, m["code_layers"], m["heads"], m["code_len"], m["dropout"])
        if flags.code != "none":
            self.idiom = nn.Linear(len(IDIOM_NAMES), d)
            self.code_proj = nn.Linear(cd + d if flags.code == "transformer" else d, d)
        if flags.use_graph:
            self.layers = nn.ModuleList(HeteroTemporalAttention(d, m["heads"], 2 * N_REL, len(EDGE_FEATS), 32,
                                                                m["dropout"], flags) for _ in range(m["layers"]))
        if flags.use_motifs:
            self.motif = nn.Sequential(nn.Linear(len(MOTIF_NAMES), d), nn.GELU(), nn.Linear(d, d))
        head_in = 3 * d + len(AUTH_FEATS) + (d if flags.code != "none" else 0) + (d if flags.use_motifs else 0)
        self.readout = nn.Sequential(nn.Linear(head_in, d), nn.GELU(), nn.Dropout(m["dropout"]), nn.Linear(d, d))
        # prediction always comes from a standard head; the prototype layer runs alongside it
        # as the explanation / novelty module (it is trained jointly but does not gate accuracy)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(d, 1))
        if flags.use_proto:
            self.proto = PrototypeLayer(d, m["prototypes_per_class"])

    def encode_code(self, tokens, idioms, cached=None):
        parts = [self.idiom(idioms)]
        if self.flags.code == "transformer":
            # the pre-trained encoder is frozen; its embeddings normally arrive precomputed (`cached`)
            parts.insert(0, cached if cached is not None else self.code_enc(tokens))
        return self.code_proj(torch.cat(parts, 1))

    def forward(self, b: dict, edge_weight=None) -> dict:
        h = self.node_in(b["x"]) + self.role(b["role"]) + self.kind(b["is_contract"])
        code = None
        if self.flags.code != "none" and b["code_tokens"].shape[0]:
            code = self.encode_code(b["code_tokens"], b["code_idioms"], b.get("code_emb"))
            slot = b["code_slot"]
            has = slot >= 0
            h = h.clone()
            h[has] = h[has] + code[slot[has]]
        if self.flags.use_graph:
            src = torch.cat([b["esrc"], b["edst"]])
            dst = torch.cat([b["edst"], b["esrc"]])
            rel = torch.cat([b["etype"], b["etype"] + N_REL])
            tf = self.tenc(torch.cat([b["edt"], b["edt"]]))
            ef = torch.cat([b["efeat"], b["efeat"]])
            ew = None if edge_weight is None else torch.cat([edge_weight, edge_weight])
            for layer in self.layers:
                h = layer(h, src, dst, rel, tf, ef, ew)
        c = b["centres"]
        parts = [h[c[:, 0]], h[c[:, 1]], h[c[:, 2]], b["authf"]]
        if self.flags.code != "none":
            cs = b["code_slot"][c[:, 2]]
            cvec = torch.zeros(len(c), h.shape[1], device=h.device)
            if code is not None:
                cvec[cs >= 0] = code[cs[cs >= 0]]
            parts.append(cvec)
        if self.flags.use_motifs:
            parts.append(self.motif(b["motifs"]))
        z = self.readout(torch.cat(parts, 1))
        out = dict(logit=self.out(z).squeeze(1), z=z)
        if self.flags.use_proto:
            plogit, sim, d2 = self.proto(z)
            out.update(proto_logit=plogit, sim=sim, d2=d2)
        return out
