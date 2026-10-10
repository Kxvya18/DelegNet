"""Paper figures (monochrome, IEEE column width) and CSV tables from the experiment outputs.

  source .env.local; PYTHONPATH=. .venv/bin/python results/make_figures.py
      # writes results/figures/*.pdf|png and results/tables/*.csv
"""
import glob
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from results.paper_names import name, paper_only as _paper_only  # noqa: E402

RUNS = Path("runs")
DATA = Path(os.environ.get("DELEGNET_DATA_ROOT", "data/full")) / "ethereum"
SCAN = Path(os.environ.get("DELEGNET_SCAN_ROOT", "data")) / "ethereum"
FIG, TAB = Path("results/figures"), Path("results/tables")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ORDER = ["DelegNet", "TGAT", "GraphSAGE", "XGBoost", "Fuzzy hash", "Blocklist"]
STYLE = {  # monochrome: grey level, hatch, marker, line style
    "DelegNet":   ("0.00", "",     "o", "-"),
    "TGAT":       ("0.50", "\\\\", "^", "-."),
    "GraphSAGE":  ("0.65", "xx",   "v", ":"),
    "XGBoost":    ("0.80", "..",   "D", (0, (5, 1))),
    "Fuzzy hash": ("0.90", "--",   "P", (0, (3, 1, 1, 1))),
    "Blocklist":  ("1.00", "oo",   "X", (0, (1, 2))),
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.visible": True,
    "ytick.minor.visible": True, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "0.0", "legend.fancybox": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "hatch.linewidth": 0.5,
})
COL, DCOL = 3.5, 7.16   # IEEE single / double column width (in)


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png")
    plt.close(fig)


def bars(ax, cats, methods, mean, err, ylim=None, ylabel=""):
    n = len(methods)
    w = 0.8 / n
    x = np.arange(len(cats))
    for i, m in enumerate(methods):
        g, h, _, _ = STYLE[m]
        ax.bar(x - 0.4 + w * (i + 0.5), mean[m], w, yerr=err[m], color=g, hatch=h, edgecolor="0", linewidth=0.5,
               error_kw=dict(elinewidth=0.5, capsize=1.2, capthick=0.5), label=m)
    ax.set_xticks(x, cats)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.85", linewidth=0.4)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ 1. zero-day accuracy by slice
z = pd.read_csv(RUNS / "zeroday/metrics.csv")
z = z[z.seed >= 0].assign(model=lambda d: d.method.map(name)).pipe(_paper_only)
slices = [("all", "All delegations"), ("first10_per_contract", "First 10 victims"),
          ("first1_per_contract", "First victim")]
tab = z.groupby(["model", "split"])[["auroc", "auprc", "f1", "mcc", "recall_at_fpr"]].agg(["mean", "std"]).round(4)
tab.to_csv(TAB / "zeroday_by_slice.csv")
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.1), gridspec_kw=dict(wspace=0.22))
for ax, metric, lab, yl in ((axes[0], "auroc", "AUROC", (0.45, 1.01)), (axes[1], "auprc", "AUPRC", (0.75, 1.005))):
    piv = z.pivot_table(index="model", columns="split", values=metric, aggfunc=["mean", "std"])
    mean = {m: [piv["mean"].loc[m, s] for s, _ in slices] for m in ORDER}
    err = {m: [piv["std"].loc[m, s] for s, _ in slices] for m in ORDER}
    bars(ax, [t for _, t in slices], ORDER, mean, err, yl, lab)
axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02, 2.2, 0.2), mode="expand", borderaxespad=0)
save(fig, "fig_zeroday_slices")

# ------------------------------------------------------------------ 3. adaptive-attacker evasion suite
e = pd.read_csv(RUNS / "evasion_suite.csv").assign(model=lambda d: d.method.map(name)).pipe(_paper_only)
attacks = [("original", "None"), ("self_submit", "Self-\nsubmit"), ("fresh_sponsor", "Fresh\nsponsor"),
           ("fresh_contract", "Fresh\ncontract"), ("code_disguise", "Code\ndisguise"), ("combined", "Combined")]
e.groupby(["model", "setting"])[["auroc", "recall_at_fpr"]].agg(["mean", "std"]).round(4).to_csv(TAB / "evasion_suite.csv")
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.2), gridspec_kw=dict(wspace=0.22))
for ax, metric, lab, yl in ((axes[0], "recall_at_fpr", "Recall at 1% FPR", (0, 1.05)),
                            (axes[1], "auroc", "AUROC", (0.45, 1.01))):
    piv = e.pivot_table(index="model", columns="setting", values=metric, aggfunc=["mean", "std"])
    meth = [m for m in ORDER if m in piv.index]
    bars(ax, [t for _, t in attacks], meth, {m: [piv["mean"].loc[m, a] for a, _ in attacks] for m in meth},
         {m: [piv["std"].loc[m, a] for a, _ in attacks] for m in meth}, yl, lab)
axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02, 2.2, 0.2), mode="expand", borderaxespad=0)
save(fig, "fig_evasion_suite")

# ------------------------------------------------------------------ dense figures from full predictions
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from sklearn.metrics import roc_auc_score as roc_auc_score_  # noqa: E402

pr_all = pd.read_parquet(RUNS / "zeroday/predictions.parquet")
pr_all = pr_all[pr_all.seed >= 0].assign(model=lambda d: d.method.map(name)).pipe(_paper_only)


def curves(ax_roc, ax_pr, df, title):
    for m in ORDER:
        d = df[df.model == m]
        if d.y.nunique() < 2:
            continue
        g, _, mk, ls = STYLE[m]
        col = "0" if g == "1.00" else g
        fpr, tpr, _ = roc_curve(d.y, d.p)
        ax_roc.plot(np.maximum(fpr, 1e-4), tpr, color=col, ls=ls, lw=1.1 if m == "DelegNet" else 0.8, label=m,
                    marker=mk, markevery=max(len(fpr) // 8, 1), mfc="white" if m != "DelegNet" else "0", mec="0", ms=2.8)
        prec, rec, _ = precision_recall_curve(d.y, d.p)
        ax_pr.plot(rec, prec, color=col, ls=ls, lw=1.1 if m == "DelegNet" else 0.8,
                   marker=mk, markevery=max(len(rec) // 8, 1), mfc="white" if m != "DelegNet" else "0", mec="0", ms=2.8)
    ax_roc.plot([1e-4, 1], [1e-4, 1], color="0.7", lw=0.5, ls=":")
    ax_roc.set_xscale("log")
    ax_roc.set_xlim(1e-4, 1)
    ax_roc.set_ylim(0, 1.01)
    ax_roc.set_xlabel("False-positive rate (log)")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_title(title, loc="left")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_ylim(0.75, 1.005)
    for a in (ax_roc, ax_pr):
        a.grid(color="0.88", linewidth=0.4, which="major")


fig, axes = plt.subplots(2, 3, figsize=(DCOL, 4.3), gridspec_kw=dict(wspace=0.32, hspace=0.45))
for col, (sl, title) in enumerate((("all", "(a) all delegations"), ("first10", "(b) first 10 victims"),
                                   ("first1", "(c) first victim"))):
    df = pr_all if sl == "all" else pr_all[pr_all.rank_in_contract <= (10 if sl == "first10" else 1)]
    curves(axes[0, col], axes[1, col], df, title)
axes[0, 0].legend(ncol=7, loc="lower left", bbox_to_anchor=(0, 1.12, 3.75, 0.2), mode="expand", borderaxespad=0)
save(fig, "fig_roc_pr")

# score distributions on first victims: malicious (left half) vs benign (right half) per method
fig, ax = plt.subplots(figsize=(DCOL, 2.0))
f1 = pr_all[pr_all.rank_in_contract <= 10]
for i, m in enumerate(ORDER):
    for side, cls in ((-1, 1), (1, 0)):
        v = f1[(f1.model == m) & (f1.y == cls)].p.values
        if len(v) < 3:
            continue
        kde_x = np.linspace(0, 1, 200)
        from scipy.stats import gaussian_kde
        try:
            dens = gaussian_kde(v, bw_method=0.15)(kde_x)
        except Exception:
            continue
        dens = 0.42 * dens / dens.max()
        ax.fill_betweenx(kde_x, i, i + side * dens, color="0.25" if cls == 1 else "0.8", lw=0.4, edgecolor="0")
        q = np.quantile(v, [0.25, 0.5, 0.75])
        ax.plot([i + side * 0.05] * 2, q[[0, 2]], color="0", lw=1.2)
        ax.plot(i + side * 0.05, q[1], marker="o", ms=2.5, color="white", mec="0", mew=0.6)
ax.set_xticks(range(len(ORDER)), ORDER)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylabel("Predicted maliciousness")
ax.set_ylim(-0.02, 1.02)
ax.grid(axis="y", color="0.88", linewidth=0.4)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(fc="0.25", ec="0", label="Malicious"), Patch(fc="0.8", ec="0", label="Benign")],
          loc="lower right", ncol=2)
save(fig, "fig_score_distributions")

# history: AUROC and AUPRC by evidence bucket, per-seed points, mean line, and support bars
b = pd.read_csv(RUNS / "diagnostics/buckets.csv")
b = b[~b.method.str.contains("ensemble")].assign(model=lambda d: d.method.map(name)).pipe(_paper_only)
b.groupby(["model", "bucket"])[["n", "pos", "neg", "auroc", "auprc"]].mean().round(4).to_csv(TAB / "history_buckets.csv")
order_b = ["1st victim", "victims 2-10", "victims 11-100", "victims >100"]
fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.1), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1, 0.75]))
for ax, metric, yl in ((axes[0], "auroc", (0.45, 1.02)), (axes[1], "auprc", (0.75, 1.005))):
    for k, m in enumerate(ORDER):
        g, _, mk, ls = STYLE[m]
        col = "0" if g == "1.00" else g
        d = b[b.model == m]
        mean = d.groupby("bucket")[metric].mean().reindex(order_b)
        lo = d.groupby("bucket")[metric].min().reindex(order_b)
        hi = d.groupby("bucket")[metric].max().reindex(order_b)
        x = np.arange(4) + (k - 3) * 0.035
        if m == "DelegNet":
            ax.fill_between(x, lo, hi, color="0.85", lw=0)
        ax.plot(x, mean, color=col, ls=ls, marker=mk, mfc="white" if m != "DelegNet" else "0", mec="0",
                lw=1.1 if m == "DelegNet" else 0.8, label=m)
        for xi, bk in zip(x, order_b):
            pts = d[d.bucket == bk][metric].values
            ax.scatter(np.full(len(pts), xi), pts, s=4, color=col, edgecolor="none", alpha=0.7)
    ax.set_xticks(range(4), ["1st", "2–10", "11–100", ">100"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(*yl)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Victim position")
    ax.grid(color="0.88", linewidth=0.4)
sup = b[b.model == "DelegNet"].groupby("bucket")[["pos", "neg"]].mean().reindex(order_b)
axes[2].bar(range(4), sup.pos, color="0.25", edgecolor="0", lw=0.5, label="Malicious")
axes[2].bar(range(4), sup.neg, bottom=sup.pos, color="0.85", edgecolor="0", lw=0.5, hatch="///", label="Benign")
axes[2].set_yscale("log")
axes[2].set_xticks(range(4), ["1st", "2–10", "11–100", ">100"])
axes[2].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[2].set_ylabel("Test delegations per fold")
axes[2].set_xlabel("Victim position")
axes[2].legend(fontsize=6)
axes[0].legend(ncol=7, loc="lower left", bbox_to_anchor=(0, 1.04, 3.1, 0.2), mode="expand", borderaxespad=0)
save(fig, "fig_history_buckets")

# code disguise: performance under increasing mutation and the mechanism behind it
r = pd.read_csv(RUNS / "robustness_code.csv").assign(model=lambda d: d.method.map(name)).pipe(_paper_only)
r.groupby(["model", "strength"])[["auroc", "recall_at_fpr"]].agg(["mean", "std"]).round(4).to_csv(TAB / "code_disguise.csv")
import random as _random  # noqa: E402
import pickle as _pickle  # noqa: E402
from delegnet.evm import mutate_bytecode, opcode_sequence  # noqa: E402
from delegnet.patterns import code_idioms  # noqa: E402

with open(DATA / "graph.pkl", "rb") as fh:
    G = _pickle.load(fh)
tgt = pd.read_parquet(DATA / "targets.parquet")
mal_nodes = sorted(set(tgt[tgt.label == 1].contract_node) & set(G.code_bytes))[:60]


def jac(a, b):
    A, B = set(zip(a, a[1:], a[2:], a[3:])), set(zip(b, b[1:], b[2:], b[3:]))
    return len(A & B) / max(len(A | B), 1)


rows = []
rng = _random.Random(0)
for n in mal_nodes:
    raw = G.code_bytes[n]
    for st in (0.25, 0.5, 1, 2, 3, 4):
        mut = mutate_bytecode(raw, rng, st)
        rows.append(dict(contract=n, strength=st, raw=jac(opcode_sequence(raw), opcode_sequence(mut)),
                         canonical=jac(code_idioms.tokens(raw), code_idioms.tokens(mut))))
sim = pd.DataFrame(rows)
sim.groupby("strength")[["raw", "canonical"]].describe().round(4).to_csv(TAB / "disguise_similarity.csv")
fig, axes = plt.subplots(2, 2, figsize=(DCOL, 4.0), gridspec_kw=dict(wspace=0.28, hspace=0.42))
ax = axes[0, 0]
for m in ORDER:
    d = r[r.model == m].groupby("strength").recall_at_fpr.agg(["mean", "min", "max"])
    if not len(d):
        continue
    g, _, mk, ls = STYLE[m]
    col = "0" if g == "1.00" else g
    if m == "DelegNet":
        ax.fill_between(d.index, d["min"], d["max"], color="0.85", lw=0)
    ax.plot(d.index, d["mean"], color=col, ls=ls, marker=mk, mfc="white" if m != "DelegNet" else "0", mec="0",
            lw=1.1 if m == "DelegNet" else 0.8, label=m)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("Mutation strength")
ax.set_ylabel("Recall at 1% FPR")
ax.set_title("(a) detection under disguise", loc="left")
ax.legend(ncol=4, fontsize=5.4, loc="lower left", bbox_to_anchor=(0.0, 0.08))
for ax, colm, title in ((axes[0, 1], "raw", "(b) raw opcode similarity (fuzzy hash)"),
                        (axes[1, 0], "canonical", "(c) canonical similarity (DelegNet)")):
    for n, d in sim.groupby("contract"):
        ax.plot(d.strength, d[colm], color="0.6", lw=0.25, alpha=0.6)
    q = sim.groupby("strength")[colm].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    ax.fill_between(q.index, q[0.1], q[0.9], color="0.88", lw=0, zorder=0)
    ax.fill_between(q.index, q[0.25], q[0.75], color="0.72", lw=0, zorder=0)
    ax.plot(q.index, q[0.5], color="0", lw=1.3, marker="o", ms=3, label="median")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mutation strength")
    ax.set_ylabel("4-gram similarity to original")
    ax.set_title(title, loc="left")
ax = axes[1, 1]
for k, st in enumerate(sorted(sim.strength.unique())):
    d = sim[sim.strength == st]
    ax.scatter(d.raw, d.canonical, s=7, marker="osD^vP"[k], facecolor="white" if k % 2 else "0.3", edgecolor="0",
               lw=0.4, label=f"strength {st:g}")
ax.plot([0, 1], [0, 1], color="0.5", lw=0.6, ls="--")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Raw similarity")
ax.set_ylabel("Canonical similarity")
ax.set_title("(d) per-contract: canonical vs raw", loc="left")
ax.legend(ncol=2, fontsize=5.4, loc="lower right")
for a_ in axes.flat:
    a_.grid(color="0.88", linewidth=0.4)
save(fig, "fig_code_disguise")

# evasion: parallel-coordinate profiles across attacks (recall and AUROC), per-seed points
e2 = e.pivot_table(index="model", columns="setting", values="auroc", aggfunc="mean")
e2[[a for a, _ in attacks if a != "original"]].sub(e2["original"], axis=0).round(4).to_csv(TAB / "evasion_auroc_drop.csv")
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.3), gridspec_kw=dict(wspace=0.22))
xs = np.arange(len(attacks))
for ax, metric, lab, yl in ((axes[0], "recall_at_fpr", "Recall at 1% FPR", (-0.03, 1.05)),
                            (axes[1], "auroc", "AUROC", (0.45, 1.01))):
    for k, m in enumerate(ORDER):
        d = e[e.model == m]
        if not len(d):
            continue
        g, _, mk, ls = STYLE[m]
        col = "0" if g == "1.00" else g
        mean = d.groupby("setting")[metric].mean().reindex([a for a, _ in attacks])
        jit = (k - 3) * 0.05
        ax.plot(xs + jit, mean.values, color=col, ls=ls, lw=1.3 if m == "DelegNet" else 0.8, marker=mk,
                mfc="0" if m == "DelegNet" else "white", mec="0", ms=3.5, label=m, zorder=3 if m == "DelegNet" else 2)
        for j, (a_, _) in enumerate(attacks):
            pts = d[d.setting == a_][metric].values
            ax.scatter(np.full(len(pts), j + jit), pts, s=5, color=col, edgecolor="0", lw=0.2, zorder=1)
    for x in xs:
        ax.axvline(x, color="0.85", lw=0.5, zorder=0)
    ax.set_xticks(xs, [t for _, t in attacks], fontsize=6)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(*yl)
    ax.set_ylabel(lab)
axes[0].legend(ncol=7, loc="lower left", bbox_to_anchor=(0, 1.03, 2.25, 0.2), mode="expand", borderaxespad=0)
save(fig, "fig_evasion_profiles")

# pre-signature screening: ROC / PR at signing time, recomputed from the saved zero-day models
p = pd.read_csv(RUNS / "presignature.csv").assign(model=lambda d: d.method.map(name)).pipe(_paper_only)
p.groupby("model")[["auroc", "recall_at_fpr", "false_alarm"]].agg(["mean", "std"]).round(4).to_csv(TAB / "presignature.csv")
cache = TAB / "presignature_scores.csv"
if not cache.exists():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-W", "ignore", "results/presignature_scores.py"], check=True)
ps_sc = pd.read_csv(cache).assign(model=lambda d: d.method.map(name)).pipe(_paper_only)
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.2), gridspec_kw=dict(wspace=0.28))
curves(axes[0], axes[1], ps_sc, "Screening at signing time")
axes[1].set_ylim(0.6, 1.005)
axes[0].legend(ncol=7, loc="lower left", bbox_to_anchor=(0, 1.1, 2.25, 0.2), mode="expand", borderaxespad=0)
save(fig, "fig_presignature")

# contract level: per-contract mean score over the first 10 victims (seed 0) and bootstrap AUROC
cl = pr_all[(pr_all.seed == 0) & (pr_all.rank_in_contract <= 10)].groupby(["model", "contract_node"]).agg(
    y=("y", "first"), p=("p", "mean")).reset_index()
mat = cl.pivot_table(index="contract_node", columns="model", values="p")[[m for m in ORDER if m in set(cl.model)]]
lab_c = cl.groupby("contract_node").y.first().loc[mat.index]
mat = mat.loc[lab_c.sort_values(ascending=False).index]
mat = pd.concat([mat[lab_c.loc[mat.index] == 1].sort_values("DelegNet", ascending=False),
                 mat[lab_c.loc[mat.index] == 0].sort_values("DelegNet", ascending=False)])
rng_b = np.random.default_rng(0)
boot = {}
for m in mat.columns:
    yv, pv = lab_c.loc[mat.index].values, mat[m].values
    vals = []
    for _ in range(2000):
        i = rng_b.integers(0, len(yv), len(yv))
        if 0 < yv[i].sum() < len(i):
            vals.append(roc_auc_score_(yv[i], pv[i]))
    boot[m] = np.array(vals)
pd.DataFrame({m: [np.mean(v), np.quantile(v, .025), np.quantile(v, .975)] for m, v in boot.items()},
             index=["auroc_mean", "ci_lo", "ci_hi"]).T.round(4).to_csv(TAB / "contract_level.csv")
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 3.0), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1.2]))
im = axes[0].imshow(mat.values, cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
nm = int((lab_c.loc[mat.index] == 1).sum())
axes[0].axhline(nm - 0.5, color="0", lw=1.0)
axes[0].text(len(mat.columns) - 0.4, nm / 2, "malicious", rotation=270, va="center", fontsize=6)
axes[0].text(len(mat.columns) - 0.4, nm + (len(mat) - nm) / 2, "benign", rotation=270, va="center", fontsize=6)
axes[0].set_xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right", fontsize=6)
axes[0].set_yticks([])
axes[0].set_ylabel(f"{len(mat)} held-out contracts")
axes[0].tick_params(which="both", length=0)
axes[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
cb = fig.colorbar(im, ax=axes[0], fraction=0.05, pad=0.08)
cb.set_label("Mean score, first 10 victims", labelpad=2)
ms_ = list(boot)
vp = axes[1].violinplot([boot[m] for m in ms_], positions=range(len(ms_)), vert=False, widths=0.8,
                        showextrema=False)
for body, m in zip(vp["bodies"], ms_):
    body.set_facecolor("0.3" if m == "DelegNet" else "0.85")
    body.set_edgecolor("0")
    body.set_linewidth(0.4)
    body.set_alpha(1)
for i, m in enumerate(ms_):
    lo, hi = np.quantile(boot[m], [.025, .975])
    axes[1].plot([lo, hi], [i, i], color="0", lw=0.8)
    axes[1].plot(np.median(boot[m]), i, marker="|", ms=8, color="white" if m == "DelegNet" else "0", mew=1.2)
axes[1].set_yticks(range(len(ms_)), ms_)
axes[1].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[1].invert_yaxis()
axes[1].axvline(0.5, color="0.5", lw=0.5, ls=":")
axes[1].set_xlabel("Contract-level AUROC (2,000 bootstrap resamples)")
axes[1].grid(axis="x", color="0.88", linewidth=0.4)
save(fig, "fig_contract_level")

# leakage audit: (a) test-label permutation null vs observed AUROC; (b) retraining on shuffled labels
f0 = pr_all[(pr_all.seed == 0) & (pr_all.split == "fold0")]
rng_p = np.random.default_rng(1)
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.4), gridspec_kw=dict(wspace=0.35, width_ratios=[1.4, 1]))
mods = [m for m in ORDER if m in set(f0.model)]
null_rows = []
for i, m in enumerate(mods):
    d = f0[f0.model == m]
    obs = roc_auc_score_(d.y, d.p)
    null = np.array([roc_auc_score_(rng_p.permutation(d.y.values), d.p.values) for _ in range(1000)])
    null_rows.append(dict(model=m, observed=obs, null_mean=null.mean(), null_q975=np.quantile(null, .975),
                          p_value=(np.sum(null >= obs) + 1) / (len(null) + 1)))
    v = axes[0].violinplot(null, positions=[i], vert=False, widths=0.75, showextrema=False)
    for body in v["bodies"]:
        body.set_facecolor("0.85")
        body.set_edgecolor("0")
        body.set_linewidth(0.4)
        body.set_alpha(1)
    axes[0].plot(obs, i, marker="*", ms=7, color="0", ls="none")
    axes[0].plot([np.quantile(null, .025), np.quantile(null, .975)], [i, i], color="0.3", lw=0.6)
pd.DataFrame(null_rows).round(4).to_csv(TAB / "permutation_test.csv", index=False)
axes[0].set_yticks(range(len(mods)), mods)
axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[0].invert_yaxis()
axes[0].axvline(0.5, color="0.5", lw=0.5, ls=":")
axes[0].set_xlim(0.3, 1.02)
axes[0].set_xlabel("Test AUROC: label-permutation null (violin) vs observed (star)")
ls_ = pd.read_csv(RUNS / "diagnostics/label_shuffle.csv").assign(model=lambda d: d.method.map(name)).pipe(_paper_only)
ls_.to_csv(TAB / "label_shuffle.csv", index=False)
real_all = z[z.split == "all"].groupby("model").auroc.mean()
real_f1 = z[z.split == "first1_per_contract"].groupby("model").auroc.mean()
sm = sorted(ls_.model.unique())
for i, m in enumerate(sm):
    for off, colm, real, mk in ((-0.15, "test_auroc_shuffled", real_all, "o"), (0.15, "first1_auroc_shuffled", real_f1, "s")):
        v = ls_[ls_.model == m][colm].dropna().values
        axes[1].scatter(v, np.full(len(v), i + off), s=12, marker=mk, facecolor="white", edgecolor="0", lw=0.6)
        axes[1].plot(real.get(m, np.nan), i + off, marker=mk, ms=5, color="0", ls="none")
        axes[1].plot([v.mean()] * 2, [i + off - 0.08, i + off + 0.08], color="0", lw=1.2)
axes[1].set_yticks(range(len(sm)), sm)
axes[1].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[1].axvline(0.5, color="0.5", lw=0.5, ls=":")
axes[1].set_xlim(0, 1.02)
axes[1].set_xlabel("Test AUROC after training on shuffled labels")
from matplotlib.lines import Line2D  # noqa: E402
axes[1].legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec="0", label="all, shuffled"),
                        Line2D([], [], marker="s", ls="none", mfc="white", mec="0", label="first victim, shuffled"),
                        Line2D([], [], marker="o", ls="none", color="0", label="true labels")],
               fontsize=5.6, loc="lower left")
for a_ in axes:
    a_.grid(axis="x", color="0.88", linewidth=0.4)
save(fig, "fig_leakage_audit")

# ------------------------------------------------------------------ 8. capability radar-free summary heat table
cap = pd.DataFrame({
    "Zero-day AUROC": z[z.split == "all"].groupby("model").auroc.mean(),
    "First-victim AUROC": z[z.split == "first1_per_contract"].groupby("model").auroc.mean(),
    "Combined-evasion recall": e[e.setting == "combined"].groupby("model").recall_at_fpr.mean(),
    "Self-submit recall": e[e.setting == "self_submit"].groupby("model").recall_at_fpr.mean(),
    "Disguised-code recall": r[r.strength == r.strength.max()].groupby("model").recall_at_fpr.mean(),
    "Pre-signature recall": p.groupby("model").recall_at_fpr.mean(),
}).reindex([m for m in ORDER])
cap.round(4).to_csv(TAB / "capability_summary.csv")
fig, ax = plt.subplots(figsize=(DCOL, 1.9))
im = ax.imshow(cap.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
for i in range(cap.shape[0]):
    for j in range(cap.shape[1]):
        v = cap.values[i, j]
        ax.text(j, i, "–" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=6.5,
                color="white" if (not np.isnan(v) and v > 0.6) else "black",
                fontweight="bold" if not np.isnan(v) and v == np.nanmax(cap.values[:, j]) else "normal")
ax.set_xticks(range(cap.shape[1]), [c.replace(" ", "\n", 1) for c in cap.columns])
ax.set_yticks(range(cap.shape[0]), cap.index)
ax.tick_params(which="both", length=0)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
save(fig, "fig_capability_matrix")

# ------------------------------------------------------------------ 9. dataset characterisation
auth = pd.concat([pd.read_parquet(f, columns=["block", "ts", "delegate", "authority", "batch_size", "revoke"])
                  for f in glob.glob(str(SCAN / "auth/*.parquet"))], ignore_index=True)
labs = pd.read_parquet(DATA / "contract_labels.parquet")
dl = pd.read_parquet(DATA / "delegations.parquet")
auth["day"] = pd.to_datetime(auth.ts, unit="s").dt.floor("D")
lab_map = labs.label.to_dict()
auth["cls"] = auth.delegate.map(lab_map).fillna(-1)
fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.0), gridspec_kw=dict(wspace=0.3))
daily = auth[~auth.revoke].groupby(["day", "cls"]).size().unstack(fill_value=0)
for cls, (lab, ls) in {1: ("Malicious", "-"), 0: ("Benign", "--"), -1: ("Unlabelled", ":")}.items():
    if cls in daily:
        axes[0].plot(daily.index, daily[cls], color="0", ls=ls, label=lab)
axes[0].set_yscale("log")
axes[0].set_ylabel("Authorizations per day")
import matplotlib.dates as mdates
axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
axes[0].xaxis.set_minor_locator(mdates.DayLocator(interval=7))
axes[0].tick_params(axis="x", rotation=40)
axes[0].legend(fontsize=6)
per = auth[~auth.revoke].groupby("delegate").authority.nunique()
for cls, (lab, ls) in {1: ("Malicious", "-"), 0: ("Benign", "--"), -1: ("Unlabelled", ":")}.items():
    v = np.sort(per[per.index.map(lambda a: lab_map.get(a, -1)) == cls].values)
    if len(v):
        axes[1].plot(v, 1 - np.arange(len(v)) / len(v), color="0", ls=ls, label=lab, drawstyle="steps-post")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Distinct delegators per contract")
axes[1].set_ylabel("CCDF")
bt = auth.groupby("block").ts.first()
d = dl[(dl.label == 1) & dl.first_drain_block.notna()]
gap = np.interp(d.first_drain_block, bt.index, bt.values) - np.interp(d.block, bt.index, bt.values)
gap = np.sort(np.maximum(gap, 0) + 1)
axes[2].plot(gap, np.arange(1, len(gap) + 1) / len(gap), color="0", drawstyle="steps-post")
axes[2].set_xscale("log")
axes[2].set_xlabel("Seconds from delegation to first theft (+1)")
axes[2].set_ylabel("CDF")
for ax in axes:
    ax.grid(color="0.88", linewidth=0.4, which="major")
save(fig, "fig_dataset")
pd.DataFrame({
    "authorizations": [len(auth)], "non_revoking": [int((~auth.revoke).sum())],
    "authorities": [auth.authority.nunique()], "delegate_contracts": [auth.delegate.nunique()],
    "batched_share": [float((auth.batch_size > 1).mean())],
    "labelled_malicious_contracts": [int((labs.label == 1).sum())], "labelled_benign_contracts": [int((labs.label == 0).sum())],
    "same_block_theft_share": [float((gap <= 1).mean())], "median_theft_delay_s": [float(np.median(gap - 1))],
}).T.rename(columns={0: "value"}).to_csv(TAB / "dataset_statistics.csv")

# ------------------------------------------------------------------ preview grid for review
pngs = sorted(FIG.glob("fig_*.png"))
import matplotlib.image as mpimg  # noqa: E402
fig, axes = plt.subplots(int(np.ceil(len(pngs) / 2)), 2, figsize=(14, 4.2 * np.ceil(len(pngs) / 2)))
for ax, f in zip(axes.flat, pngs):
    ax.imshow(mpimg.imread(f))
    ax.set_title(f.stem, fontsize=11)
    ax.axis("off")
for ax in list(axes.flat)[len(pngs):]:
    ax.axis("off")
fig.savefig("results/figures_preview.png", dpi=110)
print("figures:", [f.name for f in pngs])
