"""DelegNet: evidence-gated stacking of a temporal-graph expert and a code-pattern expert.

  graph expert   TGAT-style temporal graph attention (TGAT's own tuned configuration)
  code expert    canonical-code retrieval memory (similarity to known scam / benign code, own
                 family excluded) + bytecode idioms, logistic model fit on training rows
  evidence gate  the Pattern Recognition Module's decision layer: from the temporal attack motifs
                 (how much behavioural evidence a contract already has) it learns per delegation
                 how far to trust each expert. It is fit on half of the *validation families*, i.e.
                 under the same unseen-family conditions as the test; the other half of the
                 validation families sets the decision thresholds.

Variants (ablations): gate="learned" (DelegNet), "average" (no gate), "graph" / "code" (one expert).
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import torch
import torch.nn.functional as F

from .patterns.motifs import MOTIF_NAMES

log = logging.getLogger("stacking")
STACKED = {"DelegNet": "learned", "DelegNet-noGate": "average", "DelegNet-graphOnly": "graph",
           "DelegNet-codeOnly": "code"}
EVIDENCE = [MOTIF_NAMES.index(n) for n in MOTIF_NAMES]


def _logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def _key(*sets):
    h = hashlib.md5()
    for S in sets:
        h.update(np.array([s["auth_id"] for s in S], dtype=np.int64).tobytes())
        h.update(np.array([s["delta"] for s in S], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _split_families(val, bank, seed):
    """Split validation rows by code family into a gate-fit half and a calibration half."""
    fams = np.array([bank.family(s["chain"], int(s["nodes"][s["centres"][2]])) for s in val])
    ys = np.array([s["y"] for s in val])
    rng = np.random.default_rng(seed)
    a = set()
    for cls in (0, 1):
        f = rng.permutation(np.unique(fams[ys == cls]))
        a |= set(f[: max(1, len(f) // 2)].tolist())
    m = np.isin(fams, list(a))
    if len(set(ys[m])) < 2 or len(set(ys[~m])) < 2:   # too few families: fall back to a row split
        m = rng.random(len(val)) < 0.5
    return [s for s, k in zip(val, m) if k], [s for s, k in zip(val, m) if not k]


class CodeExpert:
    """Logistic model on [scam-memory similarity, benign-memory similarity, idioms]."""

    def fit(self, train, bank, seed=0):
        X, y = self._x(train, bank), np.array([s["y"] for s in train], dtype=np.float32)
        torch.manual_seed(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.as_tensor((X - self.mu) / self.sd)
        yt = torch.as_tensor(y)
        self.w = torch.zeros(X.shape[1], requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        pos = (yt == 0).sum().clamp_min(1) / (yt == 1).sum().clamp_min(1)
        opt = torch.optim.LBFGS([self.w, self.b], max_iter=200)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(Xt @ self.w + self.b, yt, pos_weight=pos) \
                + 1e-3 * self.w.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return self

    @staticmethod
    def _x(S, bank):
        sims = np.array([bank.pattern_sims(s["chain"], int(s["nodes"][s["centres"][2]])) for s in S], np.float32)
        idi = []
        for s in S:
            got = bank.get(s["chain"], int(s["nodes"][s["centres"][2]]))
            idi.append(got[1] if got is not None else np.zeros(21, np.float32))
        return np.hstack([sims, np.array(idi, np.float32)])

    @torch.no_grad()
    def predict(self, S, bank):
        X = torch.as_tensor((self._x(S, bank) - self.mu) / self.sd)
        return torch.sigmoid(X @ self.w + self.b).numpy()


class Gate(torch.nn.Module):
    """p = sigmoid(g * graph_logit + (1 - g) * code_logit * a + c), g = sigmoid(evidence features)."""

    def __init__(self, n_ev):
        super().__init__()
        self.g = torch.nn.Linear(n_ev, 1)
        self.scale = torch.nn.Parameter(torch.ones(2))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ev, lg, lc):
        g = torch.sigmoid(self.g(ev)).squeeze(1)
        return g * self.scale[0] * lg + (1 - g) * self.scale[1] * lc + self.bias, g


def _evidence(S):
    return np.array([np.concatenate([s["motifs"][EVIDENCE], s["authf"]]) for s in S], np.float32)


_GRAPH_CACHE: dict = {}   # one graph expert per (seed, split) shared by all stacked variants in a run


class StackedDelegNet:
    """Saved DelegNet: graph expert + code expert + memory + evidence gate (predicts like a baseline)."""

    fit = None   # marks the object as baseline-like (analyses call .predict(samples, bank))

    def __init__(self, mode, graph, code, memory, gate=None, mu=None, sd=None, device="cpu"):
        self.mode, self.graph, self.code, self.memory = mode, graph, code, memory
        self.gate, self.mu, self.sd, self.device = gate, mu, sd, device

    def experts(self, S, bank, full=False):
        from .train import predict
        bank.memory, bank.sims = self.memory, {}
        g = predict(self.graph, S, bank, device=self.device)
        return (g if full else g["p"]), self.code.predict(S, bank)

    @torch.no_grad()
    def predict_full(self, S, bank):
        g, cp = self.experts(S, bank, full=True)
        gp = g["p"]
        extra = {"d2": g["d2"]} if "d2" in g else {}   # graph-expert prototypes: explanations / novelty
        if self.mode == "graph":
            return dict(p=gp, **extra)
        if self.mode == "code":
            return dict(p=cp, **extra)
        if self.mode == "average":
            return dict(p=1 / (1 + np.exp(-(_logit(gp) + _logit(cp)) / 2)), **extra)
        T = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        logit, gv = self.gate(T((_evidence(S) - self.mu) / self.sd), T(_logit(gp)), T(_logit(cp)))
        return dict(p=torch.sigmoid(logit).numpy(), gate=gv.numpy(), graph_p=gp, code_p=cp, **extra)

    def predict(self, S, bank):
        return self.predict_full(S, bank)["p"]


def fit_stacked(cfg, method, train, val, bank, seed, train_graph) -> tuple[StackedDelegNet, list]:
    """Fit DelegNet (or a stacked ablation). Returns the model and the calibration half of the
    validation rows, on which the caller chooses decision thresholds."""
    mode = STACKED[method]
    val0 = [s for s in val if s["delta"] == 0] or val
    fit_half, cal_half = _split_families(val0, bank, seed)
    key = (seed, _key(train, val))
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE.clear()
        _GRAPH_CACHE[key] = train_graph(train, val)
    graph = _GRAPH_CACHE[key]
    bank.build_memory(train)
    code = CodeExpert().fit([s for s in train if s["delta"] == 0] or train, bank, seed)
    model = StackedDelegNet(mode, graph, code, list(bank.memory), device=cfg["model"]["device"])
    if mode != "learned":
        return model, cal_half

    gp, cp = model.experts(fit_half, bank)
    ev_fit = _evidence(fit_half)
    mu, sd = ev_fit.mean(0), ev_fit.std(0) + 1e-6
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    torch.manual_seed(seed)
    gate = Gate(ev_fit.shape[1])
    y = T([s["y"] for s in fit_half])
    pos = (y == 0).sum().clamp_min(1) / (y == 1).sum().clamp_min(1)
    opt = torch.optim.Adam(gate.parameters(), lr=0.05, weight_decay=1e-3)
    ev, lg, lc = T((ev_fit - mu) / sd), T(_logit(gp)), T(_logit(cp))
    for _ in range(300):
        logit, _ = gate(ev, lg, lc)
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos)
        opt.zero_grad(); loss.backward(); opt.step()
    model.gate, model.mu, model.sd = gate.eval(), mu, sd
    return model, cal_half
