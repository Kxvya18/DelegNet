"""Paper tables (LaTeX, mean ± std over seeds) and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _pm(df, keys, cols):
    g = df.groupby(keys)[cols]
    m, s = g.mean(), g.std().fillna(0)
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.3f} $\\pm$ {b:.3f}" for a, b in zip(m[c], s[c])]
    return out


def _bold_best(tex_df, num_df, cols):
    for c in cols:
        best = num_df[c].idxmax()
        tex_df.loc[best, c] = "\\textbf{" + tex_df.loc[best, c] + "}"
    return tex_df


def make_report(cfg):
    root = Path(cfg["run_root"])
    rep = root / "report"
    rep.mkdir(exist_ok=True)
    cols = ["auroc", "auprc", "f1", "mcc", "recall_at_fpr"]

    for protocol in ("temporal", "family", "crosschain"):
        f = root / protocol / "metrics.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for split, grp in df.groupby("split"):
            num = grp.groupby("method")[cols].mean()
            tex = _bold_best(_pm(grp, "method", cols), num, cols)
            tex.to_latex(rep / f"table_{protocol}_{split}.tex", escape=False)
            num.round(4).to_csv(rep / f"table_{protocol}_{split}.csv")

    zd = root / "zeroday" / "metrics.csv"
    if zd.exists():   # headline: zero-day, per slice, mean ± std over seeds (ensembles have seed -1)
        df = pd.read_csv(zd)
        for split, grp in df.groupby("split"):
            num = grp.groupby("method")[["auroc", "auprc", "f1", "mcc", "recall_at_fpr"]].mean()
            tex = _bold_best(_pm(grp, "method", ["auroc", "auprc", "f1", "mcc", "recall_at_fpr"]), num,
                             ["auroc", "auprc", "f1", "mcc", "recall_at_fpr"])
            tex.to_latex(rep / f"table_zeroday_{split}.tex", escape=False)
            num.round(4).to_csv(rep / f"table_zeroday_{split}.csv")
    diag = root / "diagnostics"
    if diag.exists():
        for name in ("buckets", "contract_level", "significance", "gaps", "label_shuffle_summary"):
            f = diag / f"{name}.csv"
            if f.exists():
                d = pd.read_csv(f)
                keys = [k for k in ("method", "bucket", "slice", "metric", "vs") if k in d and name != "label_shuffle_summary"]
                (d.groupby(keys).mean(numeric_only=True) if keys else d).round(4).to_csv(rep / f"diag_{name}.csv")
    sig = root / "temporal" / "significance.csv"
    if sig.exists():
        s = pd.read_csv(sig)
        s.groupby("vs")[["diff", "ci_lo", "ci_hi", "p_value"]].mean().round(4).to_csv(rep / "significance_temporal.csv")

    ew = root / "early_warning.csv"
    if ew.exists():
        e = pd.read_csv(ew)
        keep = ["detected", "pre_drain_rate", "median_lead_s", "value_saved_ratio", "benign_false_alarm"]
        e.groupby("method")[keep].agg(["mean", "std"]).round(4).to_csv(rep / "early_warning_summary.csv")
        auc_cols = sorted([c for c in e.columns if c.startswith("auroc@")], key=lambda c: int(c[6:-1]))
        if auc_cols:
            fig, ax = plt.subplots(figsize=(5, 3.4))
            for method, grp in e.groupby("method"):
                if method in ("DelegNet", "XGBoost", "TGAT", "StaticGAT", "OpcodeKNN"):
                    xs = [int(c[6:-1]) for c in auc_cols]
                    ax.errorbar(xs, grp[auc_cols].mean(), yerr=grp[auc_cols].std(), marker="o", capsize=2, label=method)
            ax.set_xscale("symlog", linthresh=10)
            ax.set_xlabel("seconds after authorization")
            ax.set_ylabel("AUROC")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(rep / "fig_early_warning.pdf")

    rob = root / "robustness_code.csv"
    if rob.exists():
        r = pd.read_csv(rob)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for method, grp in r.groupby("method"):
            g = grp.groupby("strength").recall_at_fpr
            ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o", capsize=2, label=method)
        ax.set_xlabel("mutation strength")
        ax.set_ylabel("recall @ fixed FPR")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(rep / "fig_robustness_code.pdf")

    for name in ("robustness_structural", "open_set", "campaign_attribution"):
        f = root / f"{name}.csv"
        if f.exists():
            d = pd.read_csv(f)
            keys = [k for k in ("method", "setting") if k in d]
            (d.groupby(keys).mean(numeric_only=True) if keys else d.mean(numeric_only=True).to_frame().T) \
                .drop(columns="seed", errors="ignore").round(4).to_csv(rep / f"{name}_summary.csv")
    print(f"report written to {rep}")
