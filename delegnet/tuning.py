"""Equal-budget hyperparameter search, scored on validation data only.

Every tunable method gets the same number of random trials. A trial is trained on each zero-day
fold's training families and scored on that fold's *validation* families (whole families held
out from training, never the test families). The configuration with the best mean validation
AUROC (ties: lower class-balanced log-loss) is written to <run_root>/tuning/best.json and used by
every later experiment. Test data is never touched here.
"""
from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

log = logging.getLogger("tuning")

NEURAL = {"lr": [1e-3, 5e-4, 2e-4], "dim": [64, 128], "dropout": [0.1, 0.3], "layers": [1, 2],
          "weight_decay": [1e-4, 1e-3]}
SPACE = {
    "DelegNet": {**NEURAL, "flag:attention": [True, False]},   # backbone: attention vs mean aggregation
    "TGAT": NEURAL,
    "GraphSAGE": NEURAL,
    "XGBoost": {"max_depth": [3, 4, 6, 8], "learning_rate": [0.03, 0.1], "min_child_weight": [1, 5],
                "subsample": [0.7, 1.0], "colsample_bytree": [0.6, 0.9]},
}


def sample_trials(method: str, n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(f"{method}-{seed}")
    space = SPACE[method]
    seen, out = set(), []
    while len(out) < n and len(seen) < np.prod([len(v) for v in space.values()]):
        t = {k: rng.choice(v) for k, v in space.items()}
        key = json.dumps(t, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def apply(cfg: dict, method: str, params: dict) -> tuple[dict, dict]:
    """(cfg with the method's model overrides, xgboost params)."""
    if method == "XGBoost":
        return cfg, dict(params)
    cfg = copy.deepcopy(cfg)
    flags = {k.split(":", 1)[1]: v for k, v in params.items() if k.startswith("flag:")}
    cfg["model"].update({k: v for k, v in params.items() if not k.startswith("flag:")})
    if flags:
        cfg["model"].setdefault("flag_overrides", {})[method] = flags
    return cfg, {}


def tuned(cfg: dict, method: str) -> tuple[dict, dict]:
    """Apply the tuned configuration if one exists. Ablations inherit DelegNet's configuration
    (minus the component they remove), so they differ from DelegNet in one factor only."""
    f = Path(cfg["run_root"]) / "tuning" / "best.json"
    if not f.exists():
        return cfg, {}
    best = json.loads(f.read_text())
    key = "DelegNet" if method.startswith("DelegNet-") else method
    if key not in best:
        return cfg, {}
    cfg2, xgb = apply(cfg, key, best[key])
    if key != method and "flag_overrides" in cfg2["model"]:   # carry backbone choice to the ablation
        cfg2["model"]["flag_overrides"][method] = cfg2["model"]["flag_overrides"].pop("DelegNet")
    return cfg2, xgb


def _val_scores(yv, pv):
    pv = np.clip(pv, 1e-6, 1 - 1e-6)
    w = np.where(yv == 1, 0.5 / max(yv.mean(), 1e-6), 0.5 / max(1 - yv.mean(), 1e-6))
    ll = float(-(w * (yv * np.log(pv) + (1 - yv) * np.log(1 - pv))).mean())
    return (roc_auc_score(yv, pv) if 0 < yv.sum() < len(yv) else np.nan), ll


def run_tuning(cfg: dict):
    from .baselines import BASELINES
    from .experiments import zeroday_folds
    from .graph import load_graph
    from .train import CodeBank, build_samples, predict, train_model

    tcfg = cfg.get("tuning", {})
    methods, n_trials = tcfg.get("methods", list(SPACE)), tcfg.get("trials", 6)
    src = cfg["experiment"]["source_chain"]
    out = Path(cfg["run_root"]) / "tuning"
    (out / "parts").mkdir(parents=True, exist_ok=True)
    base = copy.deepcopy(cfg)
    base["model"].update(epoch_samples=tcfg.get("epoch_samples", 3000), epochs=tcfg.get("epochs", 15))

    g, tg = load_graph(cfg, src)
    tg = tg.sort_values("event")
    bank = CodeBank({src: g})
    folds = [(f, build_samples(cfg, src, tr, cfg["experiment"]["train_checkpoints_s"]),
              build_samples(cfg, src, va, [0]))
             for f, tr, va, _ in zeroday_folds(tg, cfg["experiment"].get("zeroday_folds", 3), 0)]

    rows = []
    for method in methods:
        for t, params in enumerate(sample_trials(method, n_trials)):
            for fold, train, val in folds:
                part = out / "parts" / f"{method}_t{t}_f{fold}.json"
                if part.exists():
                    rows.append(json.loads(part.read_text()))
                    continue
                c, xgb = apply(base, method, params)
                yv = np.array([s["y"] for s in val])
                if method in BASELINES:
                    pv = BASELINES[method](0, xgb).fit(train, val, bank).predict(val, bank)
                else:
                    pv = predict(train_model(c, method, train, val, bank, 0), val, bank, device=c["model"]["device"])["p"]
                auroc, ll = _val_scores(yv, pv)
                row = dict(method=method, trial=t, fold=fold, params=json.dumps(params), val_auroc=auroc, val_logloss=ll)
                part.write_text(json.dumps(row))
                rows.append(row)
                log.info("tune %s trial %d fold %d: val AUROC %.4f logloss %.4f", method, t, fold, auroc, ll)

    df = pd.DataFrame(rows)
    df.to_csv(out / "trials.csv", index=False)
    agg = df.groupby(["method", "trial", "params"]).agg(val_auroc=("val_auroc", "mean"),
                                                        val_logloss=("val_logloss", "mean")).reset_index()
    best = {}
    for method, grp in agg.groupby("method"):
        top = grp.sort_values(["val_auroc", "val_logloss"], ascending=[False, True]).iloc[0]
        best[method] = json.loads(top.params)
        log.info("best %s: %s (val AUROC %.4f)", method, top.params, top.val_auroc)
    agg.to_csv(out / "summary.csv", index=False)
    (out / "best.json").write_text(json.dumps(best, indent=2))
