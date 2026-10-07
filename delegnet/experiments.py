"""Evaluation protocols and analyses for the paper.

Protocols
  temporal    train on the earliest 60% of delegations, validate on the next 20%, test on
              the last 20% (the deployment setting; no future information in training)
  family      whole code families held out from training (unseen attack/wallet families);
              also provides the in-distribution test for the open-set experiment
  crosschain  train on the source chain, test zero-shot and k-shot on other chains

Thresholds are always chosen on validation data, never on test.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

from .baselines import BASELINES
from .evaluate import (best_f1_threshold, early_warning, explain_edges, fidelity, metrics, paired_bootstrap,
                       relation_importance, threshold_at_fpr)
from .graph import load_graph
from .model import VARIANTS
from .train import CodeBank, build_samples, predict, train_model

log = logging.getLogger("exp")

MAIN = ["HashBlocklist", "OpcodeKNN", "XGBoost", "GraphSAGE", "StaticGAT", "TGAT", "MLP+Code", "DelegNet"]
ABLATIONS = ["DelegNet-noProto", "DelegNet-noMotif", "DelegNet-noCode", "DelegNet-idiomsOnly",
             "DelegNet-noTime", "DelegNet-noRel", "DelegNet-noPretrain"]


def _labelled(tg):
    return tg[tg.label >= 0].copy()


def temporal_split(g, tg, split):
    t = _labelled(tg)
    ts = g.ts[t.event.values]
    q1, q2 = np.quantile(ts, [split[0], split[0] + split[1]])
    return t[ts <= q1], t[(ts > q1) & (ts <= q2)], t[ts > q2]


def family_split(tg, frac, seed):
    """Hold out code families until they cover `frac` of each class; split the rest randomly."""
    t = _labelled(tg)
    rng = np.random.default_rng(seed)
    held = set()
    for cls in (0, 1):
        sizes = t[t.label == cls].groupby("family").size()
        fams = rng.permutation(sizes.index.values)
        # never hold out the single largest family: it would empty the training class
        fams = [f for f in fams if f != sizes.idxmax()]
        cov = 0
        for f in fams:
            if cov >= frac * sizes.sum():
                break
            held.add(f)
            cov += sizes[f]
    ood = t[t.family.isin(held)]
    rest = t[~t.family.isin(held)].sample(frac=1, random_state=seed)
    n = len(rest)
    return rest.iloc[:int(.7 * n)], rest.iloc[int(.7 * n):int(.85 * n)], rest.iloc[int(.85 * n):], ood


def _fit_predict(cfg, method, train, val, tests: dict, bank, seed, out_dir):
    if method in BASELINES:
        m = BASELINES[method](seed) if method == "XGBoost" else BASELINES[method]()
        m.fit(train, val, bank)
        pv = m.predict([s for s in val if s["delta"] == 0], bank)
        preds = {k: dict(p=m.predict(v, bank)) for k, v in tests.items()}
        (out_dir / "models").mkdir(exist_ok=True, parents=True)
        pickle.dump(m, open(out_dir / "models" / f"{method}_s{seed}.pkl", "wb"))
    else:
        m = train_model(cfg, method, train, val, bank, seed)
        pv = predict(m, [s for s in val if s["delta"] == 0], bank, device=cfg["model"]["device"])["p"]
        preds = {k: predict(m, v, bank, device=cfg["model"]["device"]) for k, v in tests.items()}
        (out_dir / "models").mkdir(exist_ok=True, parents=True)
        torch.save(m, out_dir / "models" / f"{method}_s{seed}.pt")
    yv = np.array([s["y"] for s in val if s["delta"] == 0])
    taus = dict(tau_f1=best_f1_threshold(yv, pv), tau_fpr=threshold_at_fpr(yv, pv, cfg["experiment"]["target_fpr"]))
    return preds, taus


def _frame(samples, pred, method, seed, split):
    df = pd.DataFrame(dict(auth_id=[s["auth_id"] for s in samples], delta=[s["delta"] for s in samples],
                           cutoff=[s["cutoff"] for s in samples], y=[s["y"] for s in samples], p=pred["p"]))
    if "d2" in pred:
        df["novelty"] = pred["d2"].min(1)
    df["method"], df["seed"], df["split"] = method, seed, split
    return df


def run_protocol(cfg, protocol: str, methods=None):
    ex = cfg["experiment"]
    src = ex["source_chain"]
    out_dir = Path(cfg["run_root"]) / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    g, tg = load_graph(cfg, src)
    graphs = {src: g}
    methods = methods or (MAIN + ABLATIONS if protocol == "temporal" else MAIN)
    rows, frames = [], []

    for seed in ex["seeds"]:
        if protocol in ("temporal", "crosschain"):
            tr, va, te = temporal_split(g, tg, ex["split"])
            test_sets = {"test": (src, te, ex["checkpoints_s"] if protocol == "temporal" else [0])}
        else:
            tr, va, te, ood = family_split(tg, ex["heldout_family_frac"], seed)
            test_sets = {"test_id": (src, te, [0]), "test_ood": (src, ood, [0])}
        if protocol == "crosschain":
            for ch in ex["target_chains"]:
                gt, tt = load_graph(cfg, ch)
                graphs[ch] = gt
                test_sets[ch] = (ch, _labelled(tt), [0])
        bank = CodeBank(graphs)
        train = build_samples(cfg, src, tr, ex["train_checkpoints_s"])
        val = build_samples(cfg, src, va, ex["train_checkpoints_s"])
        tests = {k: build_samples(cfg, ch, t, d) for k, (ch, t, d) in test_sets.items()}
        log.info("seed %d | train %d val %d | %s", seed, len(tr), len(va), {k: len(v[1]) for k, v in test_sets.items()})

        for method in methods:
            preds, taus = _fit_predict(cfg, method, train, val, tests, bank, seed, out_dir)
            for split, samples in tests.items():
                df = _frame(samples, preds[split], method, seed, split)
                frames.append(df)
                d0 = df[df.delta == 0]
                rows.append(dict(method=method, seed=seed, split=split, **taus, **metrics(d0.y, d0.p, **taus)))
            log.info("%s seed %d: %s", method, seed, {r["split"]: round(r.get("auprc", np.nan), 4)
                                                     for r in rows[-len(tests):]})
            if protocol == "crosschain" and method in ("DelegNet", "XGBoost"):
                rows += _few_shot(cfg, method, train, val, test_sets, tests, bank, seed, out_dir, frames)

        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
        pd.concat(frames).to_parquet(out_dir / "predictions.parquet", index=False)

    _significance(out_dir, ex["seeds"])


def _few_shot(cfg, method, train, val, test_sets, tests, bank, seed, out_dir, frames):
    """k labelled examples per class from the *earliest* target delegations; test on the rest."""
    rows = []
    base = torch.load(out_dir / "models" / f"{method}_s{seed}.pt", weights_only=False) if method != "XGBoost" else None
    for ch in cfg["experiment"]["target_chains"]:
        samples = tests[ch]
        order = np.argsort([s["cutoff"] for s in samples])
        for k in cfg["experiment"]["few_shot_k"]:
            pos = [i for i in order if samples[i]["y"] == 1][:k]
            neg = [i for i in order if samples[i]["y"] == 0][:k]
            shot = set(pos + neg)
            if len(pos) < k or len(neg) < k:
                continue
            ft = [samples[i] for i in shot]
            rest = [s for i, s in enumerate(samples) if i not in shot]
            if method == "XGBoost":
                m = BASELINES["XGBoost"](seed).fit(train + ft * 10, val, bank)
                p = m.predict(rest, bank)
                pv = m.predict([s for s in val if s["delta"] == 0], bank)
            else:
                m = train_model(cfg, method, ft, ft, bank, seed, init=base, epochs=15, lr=cfg["model"]["lr"] / 10)
                p = predict(m, rest, bank)["p"]
                pv = predict(m, [s for s in val if s["delta"] == 0], bank)["p"]
            yv = np.array([s["y"] for s in val if s["delta"] == 0])
            taus = dict(tau_f1=best_f1_threshold(yv, pv), tau_fpr=threshold_at_fpr(yv, pv, cfg["experiment"]["target_fpr"]))
            y = np.array([s["y"] for s in rest])
            rows.append(dict(method=f"{method}+{k}shot", seed=seed, split=ch, **taus, **metrics(y, p, **taus)))
    return rows


def _significance(out_dir: Path, seeds):
    """Paired bootstrap of DelegNet vs every other method on identical test sets, per seed."""
    pr = pd.read_parquet(out_dir / "predictions.parquet")
    pr = pr[pr.delta == 0]
    res = []
    for (seed, split), grp in pr.groupby(["seed", "split"]):
        piv = grp.pivot_table(index="auth_id", columns="method", values="p")
        y = grp.groupby("auth_id").y.first().loc[piv.index].values
        if "DelegNet" not in piv or not 0 < y.sum() < len(y):
            continue
        for other in piv.columns.drop("DelegNet"):
            res.append(dict(seed=seed, split=split, vs=other,
                            **paired_bootstrap(y, piv["DelegNet"].values, piv[other].values, seed=int(seed))))
    pd.DataFrame(res).to_csv(out_dir / "significance.csv", index=False)


# ======================================================================= analyses

def run_analyses(cfg):
    ex = cfg["experiment"]
    src = ex["source_chain"]
    root = Path(cfg["run_root"])
    g, tg = load_graph(cfg, src)
    bank = CodeBank({src: g})
    tr, va, te = temporal_split(g, tg, ex["split"])
    out = {}

    # ---- 1. early warning (temporal protocol, every checkpoint)
    pr = pd.read_parquet(root / "temporal" / "predictions.parquet")
    met = pd.read_csv(root / "temporal" / "metrics.csv")
    ew = []
    for (method, seed), grp in pr[pr.split == "test"].groupby(["method", "seed"]):
        tau = met[(met.method == method) & (met.seed == seed) & (met.split == "test")].tau_fpr.iloc[0]
        r = early_warning(g, grp, tau)
        for d, gd in grp.groupby("delta"):
            if 0 < gd.y.sum() < len(gd):
                r[f"auroc@{int(d)}s"] = roc_auc_score(gd.y, gd.p)
        ew.append(dict(method=method, seed=seed, **r))
    pd.DataFrame(ew).to_csv(root / "early_warning.csv", index=False)

    test0 = build_samples(cfg, src, te, [0])
    val0 = build_samples(cfg, src, va, [0])
    train0 = build_samples(cfg, src, tr, [0])
    seeds = ex["seeds"][:3]

    def load(method, seed, protocol="temporal"):
        f = root / protocol / "models" / f"{method}_s{seed}"
        return pickle.load(open(f.with_suffix(".pkl"), "rb")) if f.with_suffix(".pkl").exists() else \
            torch.load(f.with_suffix(".pt"), weights_only=False)

    def scores(m, samples):
        return m.predict(samples, bank) if hasattr(m, "fit") else predict(m, samples, bank)["p"]

    # ---- 2. code-mutation robustness: attacker re-deploys mutated copies of malicious code
    rob = []
    mal_nodes = {int(s["nodes"][s["centres"][2]]) for s in test0 if s["y"] == 1}
    for method in ["HashBlocklist", "OpcodeKNN", "XGBoost", "DelegNet", "DelegNet-noPretrain", "DelegNet-idiomsOnly"]:
        for seed in seeds:
            m = load(method, seed)
            tau = met[(met.method == method) & (met.seed == seed) & (met.split == "test")].tau_fpr.iloc[0]
            for st in ex["mutation_strengths"]:
                bank.override.clear()
                bank.mutate(src, mal_nodes, st, seed=seed)
                p = scores(m, test0)
                y = np.array([s["y"] for s in test0])
                rob.append(dict(method=method, seed=seed, strength=st, auroc=roc_auc_score(y, p),
                                recall_at_fpr=float((p[y == 1] >= tau).mean())))
    bank.override.clear()
    pd.DataFrame(rob).to_csv(root / "robustness_code.csv", index=False)

    # ---- 3. structural evasion: attacker self-submits authorizations (no sponsor, no batching)
    ev = []
    test_masked = build_samples(cfg, src, te, [0], mask_sponsor=True)
    for method in ["XGBoost", "TGAT", "DelegNet", "DelegNet-noMotif"]:
        for seed in seeds:
            m = load(method, seed)
            tau = met[(met.method == method) & (met.seed == seed) & (met.split == "test")].tau_fpr.iloc[0]
            for name, samples in (("original", test0), ("self-submitted", test_masked)):
                p = scores(m, samples)
                y = np.array([s["y"] for s in samples])
                ev.append(dict(method=method, seed=seed, setting=name, auroc=roc_auc_score(y, p),
                               recall_at_fpr=float((p[y == 1] >= tau).mean())))
    pd.DataFrame(ev).to_csv(root / "robustness_structural.csv", index=False)

    # ---- 4. open-set: are unseen families further from every prototype?
    fam = pd.read_parquet(root / "family" / "predictions.parquet")
    os_rows = []
    for seed, grp in fam[fam.method == "DelegNet"].groupby("seed"):
        lab = np.r_[np.zeros((grp.split == "test_id").sum()), np.ones((grp.split == "test_ood").sum())]
        sc = np.r_[grp[grp.split == "test_id"].novelty, grp[grp.split == "test_ood"].novelty]
        conf = np.r_[grp[grp.split == "test_id"].p, grp[grp.split == "test_ood"].p]
        os_rows.append(dict(seed=seed, auroc_prototype_distance=roc_auc_score(lab, sc),
                            auroc_max_prob=roc_auc_score(lab, -np.abs(conf - 0.5))))
    pd.DataFrame(os_rows).to_csv(root / "open_set.csv", index=False)

    # ---- 5. campaign attribution and 6. prototype analysis
    camp, protos = [], []
    tmal = te[te.label == 1].set_index("auth_id")
    vmal = va[va.label == 1].set_index("auth_id")
    for method in ["DelegNet", "TGAT", "XGBoost"]:
        for seed in seeds:
            m = load(method, seed)
            emb = (lambda S: m.leaves(S, bank).astype(float)) if method == "XGBoost" \
                else (lambda S: predict(m, S, bank)["z"])
            vS = [s for s in val0 if s["y"] == 1]
            tS = [s for s in test0 if s["y"] == 1]
            if len(vS) < 3 or len(tS) < 3:
                continue
            zv, zt = emb(vS), emb(tS)
            yv = vmal.loc[[s["auth_id"] for s in vS], "campaign"].values
            yt = tmal.loc[[s["auth_id"] for s in tS], "campaign"].values
            scale = np.median(np.linalg.norm(zv - zv.mean(0), axis=1)) + 1e-9
            best = max(np.linspace(0.05, 3, 40), key=lambda th: adjusted_rand_score(
                yv, AgglomerativeClustering(None, distance_threshold=th * scale, linkage="average").fit_predict(zv)))
            pt = AgglomerativeClustering(None, distance_threshold=best * scale, linkage="average").fit_predict(zt)
            camp.append(dict(method=method, seed=seed, ari=adjusted_rand_score(yt, pt),
                             nmi=normalized_mutual_info_score(yt, pt)))
            if method == "DelegNet":
                d2 = predict(m, test0, bank)["d2"]
                near = d2.argmin(1)
                y = np.array([s["y"] for s in test0])
                fams = tg.set_index("auth_id").loc[[s["auth_id"] for s in test0], "family"].values
                for p in range(len(m.proto.protos)):
                    sel = near == p
                    src_aid = int(m.proto.source[p])
                    srow = tg[tg.auth_id == src_aid]
                    protos.append(dict(
                        seed=seed, proto=p, cls=int(m.proto.proto_class[p]), n_test=int(sel.sum()),
                        label_purity=float((y[sel] == int(m.proto.proto_class[p])).mean()) if sel.any() else np.nan,
                        family_purity=float(pd.Series(fams[sel]).value_counts(normalize=True).iloc[0]) if sel.any() else np.nan,
                        source_auth_id=src_aid,
                        source_tx=srow.tx_hash.iloc[0] if len(srow) else "",
                        source_contract=srow.delegate.iloc[0] if len(srow) else ""))
    pd.DataFrame(camp).to_csv(root / "campaign_attribution.csv", index=False)
    pd.DataFrame(protos).to_csv(root / "prototypes.csv", index=False)

    # ---- 7. explanations: which relations drive alerts, and are explanations faithful?
    m = load("DelegNet", seeds[0])
    pos = [s for s in test0 if s["y"] == 1][:200]
    mt, fid = [], []
    for i, s in enumerate(pos):
        mask, b = explain_edges(m, s, bank)
        if len(mask):
            mt.append((mask, b["etype"].cpu().numpy()))
            fid.append(fidelity(m, s, bank, mask, seed=i))
    out["relation_importance"] = relation_importance(mt)
    f = np.array(fid)
    out["fidelity_top10"] = float(f[:, 0].mean()) if len(f) else np.nan
    out["fidelity_random10"] = float(f[:, 1].mean()) if len(f) else np.nan
    (root / "explanations.json").write_text(json.dumps(out, indent=2))
    log.info("analyses done: %s", out)
