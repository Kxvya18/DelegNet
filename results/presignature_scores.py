"""Scores of every saved zero-day model (seed 0, last fold) at signing time, for the paper figures.
Separate script: the XGBoost worker process re-imports the main module, which must stay small."""
import pickle
from pathlib import Path


def main():
    # heavy imports stay inside main(): the spawned XGBoost worker re-imports this module and
    # must never load torch (OpenMP clash)
    import pandas as pd
    import torch

    from delegnet.config import load_config
    from delegnet.experiments import ANALYSIS_METHODS, zeroday_folds
    from delegnet.graph import load_graph
    from delegnet.train import CodeBank, build_samples, predict

    cfg = load_config("configs/paper.yaml")
    g, tg = load_graph(cfg, "ethereum")
    fold, _, _, te = list(zeroday_folds(tg.sort_values("event"), cfg["experiment"]["zeroday_folds"], 0))[-1]
    S = build_samples(cfg, "ethereum", te, [-1])
    bank = CodeBank({"ethereum": g})
    out = []
    for meth in ANALYSIS_METHODS:
        f = Path(cfg["run_root"]) / "zeroday/models" / f"{meth}_s0"
        if f.with_suffix(".pkl").exists():
            m = pickle.load(open(f.with_suffix(".pkl"), "rb"))
        elif f.with_suffix(".pt").exists():
            m = torch.load(f.with_suffix(".pt"), weights_only=False)
        else:
            continue
        p = m.predict(S, bank) if hasattr(m, "fit") else predict(m, S, bank)["p"]
        out.append(pd.DataFrame(dict(method=meth, y=[s["y"] for s in S], p=p)))
    pd.concat(out).to_csv("results/tables/presignature_scores.csv", index=False)


if __name__ == "__main__":
    main()
