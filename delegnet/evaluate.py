from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from .graph import N_REL
from .patterns.motifs import NONE, TRANSFER
from .train import collate


def threshold_at_fpr(y, p, fpr):
    neg = np.sort(np.asarray(p)[np.asarray(y) == 0])
    if not len(neg):
        return 0.5
    return float(neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))]) + 1e-12


def best_f1_threshold(y, p):
    cand = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    return float(max(cand, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def metrics(y, p, tau_f1, tau_fpr) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    out = dict(n=len(y), pos=int(y.sum()))
    if 0 < y.sum() < len(y):
        out.update(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p))
    pr = p >= tau_f1
    out.update(f1=f1_score(y, pr, zero_division=0), precision=precision_score(y, pr, zero_division=0),
               recall=recall_score(y, pr, zero_division=0), mcc=matthews_corrcoef(y, pr))
    pf = p >= tau_fpr
    out.update(recall_at_fpr=recall_score(y, pf, zero_division=0), fpr=float((pf & (y == 0)).sum() / max((y == 0).sum(), 1)))
    return out


def paired_bootstrap(y, p_a, p_b, metric=average_precision_score, n=2000, seed=0) -> dict:
    """H1: method A beats method B on the same test set."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            diffs.append(metric(y[i], p_a[i]) - metric(y[i], p_b[i]))
    diffs = np.array(diffs)
    return dict(diff=float(diffs.mean()), ci_lo=float(np.quantile(diffs, 0.025)),
                ci_hi=float(np.quantile(diffs, 0.975)), p_value=float((diffs <= 0).mean()))


# ------------------------------------------------------------------ early warning

def early_warning(g, preds: pd.DataFrame, tau: float) -> dict:
    """preds: one row per (auth_id, delta) with columns auth_id, delta, cutoff, p, y.

    An alert fires at the first checkpoint whose score crosses tau. For drained
    delegations we report whether it fired before the first third-party outflow, the
    lead time, and the share of the drained value that left *after* the alert, i.e.
    what a wallet that blocks on alert would have saved."""
    o = g.outcomes
    res, saved, total, leads = [], 0.0, 0.0, []
    for aid, grp in preds.sort_values("delta").groupby("auth_id"):
        hit = grp[grp.p >= tau]
        alert = int(hit.cutoff.iloc[0]) if len(hit) else None
        y = grp.y.iloc[0]
        row = dict(auth_id=aid, y=y, alerted=alert is not None)
        if y == 1 and o.third_ev[aid] != NONE:
            e, a = o.event[aid], g.src[o.event[aid]]
            evs = g.inc_ev[g.indptr[a]:g.indptr[a + 1]]
            end = o.end_ev[aid]
            out = evs[(evs > e) & (evs < end) & (g.etype[evs] == TRANSFER) & (g.src[evs] == a)]
            out = out[g.efeat[out, 3] > 0]
            v = np.expm1(g.efeat[out, 0])
            total += v.sum()
            if alert is not None:
                saved += v[out > alert].sum()
                row["pre_drain"] = bool(alert < o.third_ev[aid])
                if row["pre_drain"]:
                    leads.append(g.ts[o.third_ev[aid]] - g.ts[alert])
            else:
                row["pre_drain"] = False
        res.append(row)
    r = pd.DataFrame(res)
    mal = r[(r.y == 1) & r.get("pre_drain", pd.Series(dtype=bool)).notna()] if "pre_drain" in r else r.iloc[:0]
    return dict(
        detected=float(r[r.y == 1].alerted.astype(float).mean()) if (r.y == 1).any() else np.nan,
        pre_drain_rate=float(mal.pre_drain.astype(float).mean()) if len(mal) else np.nan,
        median_lead_s=float(np.median(leads)) if leads else np.nan,
        value_saved_ratio=float(saved / total) if total else np.nan,
        benign_false_alarm=float(r[r.y == 0].alerted.astype(float).mean()) if (r.y == 0).any() else np.nan,
    )


# ------------------------------------------------------------------ explanations

def explain_edges(model, sample, bank, steps=150, lr=0.05, size_coef=0.005, ent_coef=0.1, device="cpu"):
    """GNNExplainer-style soft edge mask for one delegation."""
    model.eval()
    b = collate([sample], bank, device)
    E = b["esrc"].shape[0]
    if E == 0:
        return np.zeros(0), b
    with torch.no_grad():
        target = (torch.sigmoid(model(b)["logit"]) >= 0.5).float()
    mask = torch.nn.Parameter(torch.full((E,), 2.0, device=device))
    opt = torch.optim.Adam([mask], lr=lr)
    for _ in range(steps):
        w = torch.sigmoid(mask)
        logit = model(b, edge_weight=w)["logit"]
        ent = -(w * torch.log(w + 1e-8) + (1 - w) * torch.log(1 - w + 1e-8)).mean()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target) + size_coef * w.sum() + ent_coef * ent
        opt.zero_grad(); loss.backward(); opt.step()
    return torch.sigmoid(mask).detach().cpu().numpy(), b


def relation_importance(masks_and_types) -> dict:
    rows = [(t, m) for mask, types in masks_and_types for t, m in zip(types, mask)]
    df = pd.DataFrame(rows, columns=["etype", "w"])
    names = {0: "AUTH", 1: "SPONSOR", 2: "REVOKE", 3: "TRANSFER"}
    return {names[k % N_REL]: float(v) for k, v in df.groupby("etype").w.mean().items()}


@torch.no_grad()
def fidelity(model, sample, bank, mask, frac=0.1, seed=0, device="cpu") -> tuple[float, float]:
    """Probability drop when removing the top-`frac` explained edges vs random edges."""
    b = collate([sample], bank, device)
    E = len(mask)
    if E == 0:
        return 0.0, 0.0
    k = max(1, int(frac * E))
    p0 = torch.sigmoid(model(b)["logit"]).item()
    w = torch.ones(E, device=device)
    w[torch.as_tensor(np.argsort(-mask)[:k])] = 1e-6
    p_top = torch.sigmoid(model(b, edge_weight=w)["logit"]).item()
    w = torch.ones(E, device=device)
    w[torch.as_tensor(np.random.default_rng(seed).choice(E, k, replace=False))] = 1e-6
    p_rand = torch.sigmoid(model(b, edge_weight=w)["logit"]).item()
    return p0 - p_top, p0 - p_rand
