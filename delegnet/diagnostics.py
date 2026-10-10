"""Overfitting / leakage diagnostics for the zero-day results (reviewer-facing).

  buckets.csv          metrics by how much history a contract already has when a delegation is scored
                       (1st victim, 2-10, 11-100, >100). On the 1st victim no sibling evidence exists.
  contract_level.csv   one score per contract (mean over its first 10 delegations), AUROC with a
                       bootstrap 95% CI, so giant campaigns cannot dominate.
  significance.csv     paired bootstrap of OpcodeKNN (the paper's DelegNet) vs every baseline per seed and slice.
  gaps.csv             AUROC on train / validation / test families for the last fold's saved models:
                       a large train-test gap is overfitting.
  label_shuffle.csv    models retrained on permuted training labels; test AUROC must collapse to ~0.5,
                       otherwise information leaks through the pipeline.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluate import paired_bootstrap

log = logging.getLogger("diagnostics")
BUCKETS = [(1, 1, "1st victim"), (2, 10, "victims 2-10"), (11, 100, "victims 11-100"), (101, 10 ** 9, "victims >100")]


def _auc(y, p):
    return roc_auc_score(y, p) if 0 < np.sum(y) < len(y) else np.nan


def _ap(y, p):
    return average_precision_score(y, p) if 0 < np.sum(y) < len(y) else np.nan


def buckets(pr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, seed), g in pr.groupby(["method", "seed"]):
        for lo, hi, name in BUCKETS:
            b = g[(g.rank_in_contract >= lo) & (g.rank_in_contract <= hi)]
            rows.append(dict(method=method, seed=seed, bucket=name, n=len(b), pos=int(b.y.sum()),
                             neg=int((1 - b.y).sum()), auroc=_auc(b.y, b.p), auprc=_ap(b.y, b.p)))
    return pd.DataFrame(rows)


def contract_level(pr: pd.DataFrame, n_boot=2000) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for (method, seed), g in pr.groupby(["method", "seed"]):
        c = g[g.rank_in_contract <= 10].groupby("contract_node").agg(y=("y", "first"), p=("p", "mean"))
        if c.y.nunique() < 2:
            continue
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(c), len(c))
            if c.y.iloc[i].nunique() == 2:
                boots.append(roc_auc_score(c.y.iloc[i], c.p.iloc[i]))
        rows.append(dict(method=method, seed=seed, contracts=len(c), malicious=int(c.y.sum()),
                         auroc=roc_auc_score(c.y, c.p), ci_lo=np.quantile(boots, 0.025), ci_hi=np.quantile(boots, 0.975)))
    return pd.DataFrame(rows)


def significance(pr: pd.DataFrame, ours="OpcodeKNN", metrics=("auroc", "auprc"), n_boot=2000) -> pd.DataFrame:
    rows = []
    fns = {"auroc": roc_auc_score, "auprc": average_precision_score}
    slices = {"all": lambda g: g, "first10": lambda g: g[g.rank_in_contract <= 10],
              "first1": lambda g: g[g.rank_in_contract == 1]}
    for seed, g in pr[pr.seed >= 0].groupby("seed"):
        for sname, f in slices.items():
            piv = f(g).pivot_table(index="auth_id", columns="method", values="p")
            y = f(g).groupby("auth_id").y.first().loc[piv.index].values
            if ours not in piv or not 0 < y.sum() < len(y):
                continue
            for other in piv.columns.drop(ours):
                ok = piv[[ours, other]].notna().all(1).values
                for metric in metrics:
                    r = paired_bootstrap(y[ok], piv[ours].values[ok], piv[other].values[ok], metric=fns[metric],
                                         n=n_boot, seed=int(seed))
                    rows.append(dict(seed=seed, slice=sname, metric=metric, vs=other, **r))
    return pd.DataFrame(rows)


def run_diagnostics(cfg: dict, shuffle_runs=(("XGBoost", 5),)):
    from .experiments import ANALYSIS_METHODS, _fit_predict, zeroday_folds
    from .graph import load_graph
    from .train import CodeBank, build_samples, predict

    root = Path(cfg["run_root"])
    out = root / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    pr = pd.read_parquet(root / "zeroday" / "predictions.parquet")
    buckets(pr).to_csv(out / "buckets.csv", index=False)
    contract_level(pr).to_csv(out / "contract_level.csv", index=False)
    significance(pr).to_csv(out / "significance.csv", index=False)
    log.info("buckets / contract-level / significance written")

    ex = cfg["experiment"]
    src = ex["source_chain"]
    g, tg = load_graph(cfg, src)
    tg = tg.sort_values("event")
    tg["rank_in_contract"] = tg.groupby("contract_node").cumcount() + 1
    bank = CodeBank({src: g})
    k = ex.get("zeroday_folds", 3)
    folds = list(zeroday_folds(tg, k, 0))

    # train / val / test gap with the saved seed-0 models (saved last for the final fold)
    fold, tr, va, te = folds[-1]
    sets = {"train": build_samples(cfg, src, tr, [0]), "val": build_samples(cfg, src, va, [0]),
            "test": build_samples(cfg, src, te, [0])}
    rows = []
    for f in sorted((root / "zeroday" / "models").glob("*_s0.*")):
        method = f.stem.rsplit("_s", 1)[0]
        if method not in ANALYSIS_METHODS:   # models of methods no longer in the pipeline
            continue
        m = pickle.load(open(f, "rb")) if f.suffix == ".pkl" else torch.load(f, weights_only=False)
        row = dict(method=method, fold=fold)
        for name, S in sets.items():
            p = m.predict(S, bank) if hasattr(m, "fit") else predict(m, S, bank, device=cfg["model"]["device"])["p"]
            row[f"{name}_auroc"] = _auc(np.array([s["y"] for s in S]), p)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "gaps.csv", index=False)
    log.info("train/val/test gaps written")

    # label-shuffle sanity check on fold 0
    fold, tr, va, te = folds[0]
    train = build_samples(cfg, src, tr, ex["train_checkpoints_s"])
    val = build_samples(cfg, src, va, ex["train_checkpoints_s"])
    test = build_samples(cfg, src, te, [0])
    # A model fit to random labels is a random function of informative features, so a single draw
    # can land far from 0.5 either way; the leakage test is that the MEAN over permutations is ~0.5.
    y = np.array([s["y"] for s in test])
    rows = []
    for method, reps in shuffle_runs:
        for r in range(reps):
            rng = np.random.default_rng(1000 + r)
            shuffled = []
            for S in (train, val):
                # balance the classes first: with ~95% scams and cloned scam code, a model fit to
                # random labels still learns "typical = majority class", a density effect rather
                # than leakage. On balanced data, permuted labels carry no information at all.
                ys = np.array([s["y"] for s in S])
                pos, neg = np.flatnonzero(ys == 1), np.flatnonzero(ys == 0)
                k = min(len(pos), len(neg))
                keep = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
                sub = [S[i] for i in keep]
                perm = rng.permutation([s["y"] for s in sub])
                shuffled.append([{**s, "y": float(v)} for s, v in zip(sub, perm)])
            preds, _ = _fit_predict(cfg, method, shuffled[0], shuffled[1], {"test": test}, bank, r, out)
            first = np.array([tg.set_index("auth_id").rank_in_contract.get(s["auth_id"], 0) == 1 for s in test])
            rows.append(dict(method=method, fold=fold, repeat=r, test_auroc_shuffled=_auc(y, preds["test"]["p"]),
                             first1_auroc_shuffled=_auc(y[first], preds["test"]["p"][first])))
            log.info("label shuffle %s #%d: test AUROC %.4f", method, r, rows[-1]["test_auroc_shuffled"])
    df = pd.DataFrame(rows)
    df.to_csv(out / "label_shuffle.csv", index=False)
    df.groupby("method")[["test_auroc_shuffled", "first1_auroc_shuffled"]].agg(["mean", "std"]).to_csv(
        out / "label_shuffle_summary.csv")
