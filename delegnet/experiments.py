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
from .evasion import ATTACKS, apply_attack
from .model import VARIANTS
from .train import CodeBank, build_samples, predict, train_model

log = logging.getLogger("exp")

MAIN = ["HashBlocklist", "OpcodeKNN-raw", "OpcodeKNN", "XGBoost", "GraphSAGE", "StaticGAT", "TGAT", "DelegNet"]
ABLATIONS = ["DelegNet-cleanGate", "DelegNet-noGate", "DelegNet-graphOnly", "DelegNet-codeOnly"]


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


def zeroday_folds(tg, k, seed):
    """Leave-family-out CV: families of each class are dealt into k folds; a fold's families are
    never seen in training. Validation uses whole held-out families too, so thresholds are honest."""
    t = _labelled(tg)
    fam_cls = t.groupby("family").label.agg(lambda s: int(s.mean() >= 0.5))
    rng = np.random.default_rng(seed)
    bins = {f: i % k for cls in (0, 1)
            for i, f in enumerate(rng.permutation(fam_cls[fam_cls == cls].index.values))}
    for fold in range(k):
        test_f = {f for f, b in bins.items() if b == fold}
        rest = t[~t.family.isin(test_f)]
        rest_f = rest.groupby("family").label.agg(lambda s: int(s.mean() >= 0.5))
        val_f = set()
        sizes = rest.groupby("family").size()
        for cls in (0, 1):
            cand = list(rng.permutation(rest_f[rest_f == cls].index.values))
            target, got = 0.15 * sizes[cand].sum(), 0
            # whole families until ~15% of this class, always leaving at least one for training
            for f in cand[:-1]:
                if got >= target:
                    break
                val_f.add(f)
                got += sizes[f]
        val = rest[rest.family.isin(val_f)]
        if val.label.nunique() < 2:    # too few families: fall back to a random validation slice
            val = rest.sample(frac=0.15, random_state=seed)
        yield fold, rest.drop(val.index), val, t[t.family.isin(test_f)]


def run_zeroday(cfg, k=None, tag="zeroday"):
    ex = cfg["experiment"]
    src = ex["source_chain"]
    out_dir = Path(cfg["run_root"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    g, tg = load_graph(cfg, src)
    tg = tg.sort_values("event")
    tg["rank_in_contract"] = tg.groupby("contract_node").cumcount() + 1
    bank = CodeBank({src: g})
    methods = ex.get(f"methods_{tag}") or ex.get("methods") or MAIN
    ex = {**ex, "seeds": ex.get(f"seeds_{tag}", ex["seeds"])}
    k = k or ex.get("zeroday_folds", 5)
    parts = out_dir / "parts"
    parts.mkdir(exist_ok=True)
    frames = []
    for seed in ex["seeds"]:
        for fold, tr, va, te in zeroday_folds(tg, k, seed):
            if te.label.nunique() < 2 and not len(te):
                continue
            todo = [m for m in methods if not (parts / f"{m}_s{seed}_f{fold}.parquet").exists()]
            frames += [pd.read_parquet(parts / f"{m}_s{seed}_f{fold}.parquet") for m in methods if m not in todo]
            if not todo:   # finished in an earlier (interrupted) run
                continue
            train = build_samples(cfg, src, tr, ex["train_checkpoints_s"])
            val = build_samples(cfg, src, va, ex["train_checkpoints_s"])
            test = build_samples(cfg, src, te, [0])
            log.info("zeroday seed %d fold %d | train %d val %d test %d (%d families held out)",
                     seed, fold, len(tr), len(va), len(te), te.family.nunique())
            for method in todo:
                preds, taus = _fit_predict(cfg, method, train, val, {"test": test}, bank, seed, out_dir)
                df = _frame(test, preds["test"], method, seed, f"fold{fold}")
                df["tau_fpr"], df["tau_f1"] = taus["tau_fpr"], taus["tau_f1"]
                df.to_parquet(parts / f"{method}_s{seed}_f{fold}.parquet", index=False)  # checkpoint
                frames.append(df)
    pr = pd.concat(frames)
    pr = pr.merge(tg[["auth_id", "rank_in_contract", "family", "contract_node"]], on="auth_id", how="left")
    # seed ensemble: the same averaging is applied to every method trained with more than one seed
    ens = []
    for method, grp in pr.groupby("method"):
        if grp.seed.nunique() > 1:
            # folds differ per seed, but every delegation is scored exactly once per seed
            e = grp.groupby("auth_id", as_index=False).agg(
                p=("p", "mean"), y=("y", "first"), tau_fpr=("tau_fpr", "mean"), tau_f1=("tau_f1", "mean"),
                delta=("delta", "first"), cutoff=("cutoff", "first"), rank_in_contract=("rank_in_contract", "first"),
                family=("family", "first"), contract_node=("contract_node", "first"))
            e["method"], e["seed"], e["split"] = f"{method} (ensemble)", -1, "pooled"
            ens.append(e)
    pr = pd.concat([pr] + ens, ignore_index=True)
    pr.to_parquet(out_dir / "predictions.parquet", index=False)
    # pooled over folds: each delegation is scored once per seed, by a model that never saw its family
    from sklearn.metrics import f1_score, matthews_corrcoef
    rows = []
    slices = {"all": lambda g: g, "first1_per_contract": lambda g: g[g.rank_in_contract == 1],
              "first10_per_contract": lambda g: g[g.rank_in_contract <= 10]}
    for (method, sd), grp in pr.groupby(["method", "seed"]):
        for name, sub in ((n, f(grp)) for n, f in slices.items()):
            if sub.y.nunique() == 2:
                m = metrics(sub.y, sub.p, 0.5, 0.5)
                hit = (sub.p >= sub.tau_fpr)
                m["recall_at_fpr"] = float(hit[sub.y == 1].mean())
                m["fpr"] = float(hit[sub.y == 0].mean())
                hf = (sub.p >= sub.tau_f1)
                m["f1"], m["mcc"] = f1_score(sub.y, hf), matthews_corrcoef(sub.y, hf)
                rows.append(dict(method=method, seed=sd, split=name, **m))
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)


def _fit_predict(cfg, method, train, val, tests: dict, bank, seed, out_dir):
    from .stacking import STACKED, fit_stacked
    from .tuning import tuned
    if method in STACKED:
        gcfg, _ = tuned(cfg, "TGAT")   # the graph expert uses TGAT's own tuned configuration
        m, cal = fit_stacked(cfg, method, train, val, bank, seed,
                             lambda tr, va: train_model(gcfg, "TGAT", tr, va, bank, seed))
        preds = {k: m.predict_full(v, bank) for k, v in tests.items()}
        v0 = [s for s in val if s["delta"] == 0] or val
        pv, yv = m.predict(v0, bank), np.array([s["y"] for s in v0])
        (out_dir / "models").mkdir(exist_ok=True, parents=True)
        pickle.dump(m, open(out_dir / "models" / f"{method}_s{seed}.pkl", "wb"))
        return preds, dict(tau_f1=best_f1_threshold(yv, pv),
                           tau_fpr=threshold_at_fpr(yv, pv, cfg["experiment"]["target_fpr"]))
    cfg, xgb_params = tuned(cfg, method)
    if method in BASELINES:
        m = BASELINES[method](seed, xgb_params) if method == "XGBoost" else BASELINES[method]()
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
    for k in ("gate", "graph_p", "code_p"):
        if k in pred:
            df[k] = pred[k]
    df["method"], df["seed"], df["split"] = method, seed, split
    return df


def run_protocol(cfg, protocol: str, methods=None):
    ex = cfg["experiment"]
    if protocol == "crosschain" and not ex.get("target_chains"):
        log.info("crosschain: no target chains configured, skipped")
        return
    src = ex["source_chain"]
    out_dir = Path(cfg["run_root"]) / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    g, tg = load_graph(cfg, src)
    graphs = {src: g}
    methods = methods or ex.get(f"methods_{protocol}") or ex.get("methods") or \
        (MAIN + ABLATIONS if protocol == "temporal" else MAIN)
    ex = {**ex, "seeds": ex.get(f"seeds_{protocol}", ex["seeds"])}
    rows, frames = [], []

    for seed in ex["seeds"]:
        if protocol in ("temporal", "crosschain"):
            tr, va, te = temporal_split(g, tg, ex["split"])
            test_sets = {"test": (src, te, ex["checkpoints_s"] if protocol == "temporal" else [0])}
            # the hard, deployment-relevant slice: contracts never seen during training
            new = te[~te.contract_node.isin(set(tr.contract_node) | set(va.contract_node))]
            if new.label.nunique() == 2:
                test_sets["test_unseen_contract"] = (src, new, [0])
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

        parts = out_dir / "parts"
        parts.mkdir(exist_ok=True)
        for method in methods:
            part = parts / f"{method}_s{seed}"
            if part.with_suffix(".parquet").exists():   # finished in an earlier (interrupted) run
                frames.append(pd.read_parquet(part.with_suffix(".parquet")))
                rows += json.loads(part.with_suffix(".json").read_text())
                continue
            preds, taus = _fit_predict(cfg, method, train, val, tests, bank, seed, out_dir)
            new_frames, new_rows = [], []
            for split, samples in tests.items():
                df = _frame(samples, preds[split], method, seed, split)
                new_frames.append(df)
                d0 = df[df.delta == 0]
                new_rows.append(dict(method=method, seed=seed, split=split, **taus, **metrics(d0.y, d0.p, **taus)))
            pd.concat(new_frames).to_parquet(part.with_suffix(".parquet"), index=False)   # checkpoint
            part.with_suffix(".json").write_text(json.dumps(new_rows, default=float))
            frames += new_frames
            rows += new_rows
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

ANALYSIS_METHODS = ["HashBlocklist", "OpcodeKNN-raw", "OpcodeKNN", "XGBoost", "GraphSAGE", "TGAT", "DelegNet"]


def run_analyses(cfg):
    """Capability analyses in the zero-day setting (unseen contract families), every method
    compared identically: for each seed, the last zero-day fold's saved models and that fold's
    validation-chosen thresholds."""
    ex = cfg["experiment"]
    src = ex["source_chain"]
    root = Path(cfg["run_root"])
    zd = root / "zeroday"
    g, tg = load_graph(cfg, src)
    tg = tg.sort_values("event")
    bank = CodeBank({src: g})
    k = ex.get("zeroday_folds", 3)
    zpr = pd.read_parquet(zd / "predictions.parquet")
    ew, rob, ev, camp, protos, pre = [], [], [], [], [], []

    def load(method, seed):
        f = zd / "models" / f"{method}_s{seed}"
        if f.with_suffix(".pkl").exists():
            return pickle.load(open(f.with_suffix(".pkl"), "rb"))
        if f.with_suffix(".pt").exists():
            return torch.load(f.with_suffix(".pt"), weights_only=False)
        return None

    def full(m, S):
        if hasattr(m, "predict_full"):
            return m.predict_full(S, bank)
        if hasattr(m, "fit"):
            return dict(p=m.predict(S, bank))
        return predict(m, S, bank, device=cfg["model"]["device"])

    def embeds(method, m, S):
        """Candidate embeddings for attribution; when a model offers several, the caller picks one
        by validation ARI (never test)."""
        if method == "XGBoost":
            return {"leaves": m.leaves(S, bank).astype(float)}
        if hasattr(m, "graph"):            # DelegNet: graph expert + its pattern evidence
            o = m.predict_full(S, bank)
            z = predict(m.graph, S, bank, device=cfg["model"]["device"])["z"]
            idi = np.stack([(bank.get(s["chain"], int(s["nodes"][s["centres"][2]])) or (None, np.zeros(21)))[1]
                            for s in S])
            mot = np.stack([s["motifs"] for s in S])
            zs = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-6)  # noqa: E731
            scores = np.stack([o.get("graph_p", o["p"]), o.get("code_p", o["p"]), o.get("gate", o["p"])], 1)
            return {"graph": z, "graph+scores": np.hstack([zs(z), zs(scores)]),
                    "graph+patterns": np.hstack([zs(z), zs(idi), zs(mot)])}
        if not hasattr(m, "fit"):
            return {"graph": predict(m, S, bank, device=cfg["model"]["device"])["z"]}
        return None

    def cluster(z, th, scale):
        return AgglomerativeClustering(None, distance_threshold=th * scale, linkage="average").fit_predict(z)

    for seed in ex["seeds"]:
        fold, tr, va, te = list(zeroday_folds(tg, k, seed))[-1]
        test0 = build_samples(cfg, src, te, [0])
        testcp = build_samples(cfg, src, te, ex["checkpoints_s"])
        masked = build_samples(cfg, src, te, [0], mask_sponsor=True)
        presig = build_samples(cfg, src, te, [-1])
        val0 = build_samples(cfg, src, va, [0])
        y0 = np.array([s["y"] for s in test0])
        mal_nodes = {int(s["nodes"][s["centres"][2]]) for s in test0 if s["y"] == 1}
        for method in ANALYSIS_METHODS:
            m = load(method, seed)
            sel = zpr[(zpr.method == method) & (zpr.seed == seed) & (zpr.split == f"fold{fold}")]
            if m is None or not len(sel):
                continue
            tau = float(sel.tau_fpr.iloc[0])

            # 1. early warning: score at every checkpoint after the authorization
            p = full(m, testcp)["p"]
            df = pd.DataFrame(dict(auth_id=[s["auth_id"] for s in testcp], delta=[s["delta"] for s in testcp],
                                   cutoff=[s["cutoff"] for s in testcp], y=[s["y"] for s in testcp], p=p))
            r = early_warning(g, df, tau)
            for d, gd in df.groupby("delta"):
                if 0 < gd.y.sum() < len(gd):
                    r[f"auroc@{int(d)}s"] = roc_auc_score(gd.y, gd.p)
            ew.append(dict(method=method, seed=seed, fold=fold, **r))

            # 2. disguised code: malicious contracts re-deployed as mutated copies
            for st in ex["mutation_strengths"]:
                bank.override.clear(); bank.raw_override.clear()
                bank.mutate(src, mal_nodes, st, seed=seed)
                pm = full(m, test0)["p"]
                rob.append(dict(method=method, seed=seed, strength=st, auroc=roc_auc_score(y0, pm),
                                recall_at_fpr=float((pm[y0 == 1] >= tau).mean())))
            bank.override.clear(); bank.raw_override.clear()

            # 3. adaptive-attacker evasion suite (malicious rows only; benign rows untouched)
            for attack in ATTACKS:
                S_att = apply_attack(attack, test0, masked, bank, src, strength=2.0, seed=seed)
                pa = full(m, S_att)["p"]
                ev.append(dict(method=method, seed=seed, setting=attack, auroc=roc_auc_score(y0, pa),
                               recall_at_fpr=float((pa[y0 == 1] >= tau).mean())))
                bank.override.clear(); bank.raw_override.clear()
            ev.append(dict(method=method, seed=seed, setting="original", auroc=roc_auc_score(y0, full(m, test0)["p"]),
                           recall_at_fpr=float((full(m, test0)["p"][y0 == 1] >= tau).mean())))

            # 3b. pre-signature screening: score before the authorization lands, submitter unknown
            pp = full(m, presig)["p"]
            yp = np.array([s["y"] for s in presig])
            pre.append(dict(method=method, seed=seed, auroc=roc_auc_score(yp, pp),
                            recall_at_fpr=float((pp[yp == 1] >= tau).mean()),
                            false_alarm=float((pp[yp == 0] >= tau).mean())))

            # 4. campaign attribution: cluster malicious delegations by attacker operation
            vS, tS = [s for s in val0 if s["y"] == 1], [s for s in test0 if s["y"] == 1]
            ev_v = embeds(method, m, vS) if len(vS) > 2 and len(tS) > 2 else None
            if ev_v is not None:
                ev_t = embeds(method, m, tS)
                cmap = tg.set_index("auth_id").campaign
                yv, yt = cmap.loc[[s["auth_id"] for s in vS]].values, cmap.loc[[s["auth_id"] for s in tS]].values
                best = None
                for name, zv in ev_v.items():     # embedding and threshold both chosen on validation
                    scale = np.median(np.linalg.norm(zv - zv.mean(0), axis=1)) + 1e-9
                    for th in np.linspace(0.05, 3, 40):
                        a = adjusted_rand_score(yv, cluster(zv, th, scale))
                        if best is None or a > best[0]:
                            best = (a, name, th)
                _, name, th = best
                zt = ev_t[name]
                pt = cluster(zt, th, np.median(np.linalg.norm(zt - zt.mean(0), axis=1)) + 1e-9)
                camp.append(dict(method=method, seed=seed, embedding=name, ari=adjusted_rand_score(yt, pt),
                                 nmi=normalized_mutual_info_score(yt, pt)))

            # 5. prototypes (DelegNet's graph expert): are they pure, real-pattern explanations?
            if method == "DelegNet" and hasattr(m.graph, "post_protos"):
                d2 = full(m, test0)["d2"]
                near = d2.argmin(1)
                fams = tg.set_index("auth_id").loc[[s["auth_id"] for s in test0], "family"].values
                for pi in range(len(m.graph.post_proto_class)):
                    sel_p = near == pi
                    c = int(m.graph.post_proto_class[pi])
                    src_aid = int(m.graph.post_proto_source[pi])
                    srow = tg[tg.auth_id == src_aid]
                    protos.append(dict(seed=seed, proto=pi, cls=c, n_test=int(sel_p.sum()),
                                       label_purity=float((y0[sel_p] == c).mean()) if sel_p.any() else np.nan,
                                       family_purity=float(pd.Series(fams[sel_p]).value_counts(normalize=True).iloc[0])
                                       if sel_p.any() else np.nan, source_auth_id=src_aid,
                                       source_tx=srow.tx_hash.iloc[0] if len(srow) else ""))
        log.info("analyses seed %d done", seed)

    pd.DataFrame(ew).to_csv(root / "early_warning.csv", index=False)
    pd.DataFrame(rob).to_csv(root / "robustness_code.csv", index=False)
    pd.DataFrame(ev).to_csv(root / "evasion_suite.csv", index=False)
    pd.DataFrame(pre).to_csv(root / "presignature.csv", index=False)
    pd.DataFrame(camp).to_csv(root / "campaign_attribution.csv", index=False)
    pd.DataFrame(protos).to_csv(root / "prototypes.csv", index=False)

    # 6. open set (family protocol): do unseen families look novel? uncertainty for every method,
    #    prototype distance where a model has prototypes
    fam = pd.read_parquet(root / "family" / "predictions.parquet")
    os_rows = []
    for (method, seed), grp in fam.groupby(["method", "seed"]):
        idm, ood = grp[grp.split == "test_id"], grp[grp.split == "test_ood"]
        if not len(idm) or not len(ood):
            continue
        lab = np.r_[np.zeros(len(idm)), np.ones(len(ood))]
        row = dict(method=method, seed=seed,
                   auroc_uncertainty=roc_auc_score(lab, -np.abs(np.r_[idm.p, ood.p] - 0.5)))
        if "novelty" in grp and grp.novelty.notna().all():
            row["auroc_prototype_distance"] = roc_auc_score(lab, np.r_[idm.novelty, ood.novelty])
        os_rows.append(row)
    pd.DataFrame(os_rows).to_csv(root / "open_set.csv", index=False)

    # 7. explanations of DelegNet's graph expert: which relations drive alerts, and are they faithful?
    out = {}
    m = load("DelegNet", ex["seeds"][0])
    if m is not None:
        fold, tr, va, te = list(zeroday_folds(tg, k, ex["seeds"][0]))[-1]
        pos = [s for s in build_samples(cfg, src, te, [0]) if s["y"] == 1][:200]
        mt, fid = [], []
        for i, s in enumerate(pos):
            mask, b = explain_edges(m.graph, s, bank)
            if len(mask):
                mt.append((mask, b["etype"].cpu().numpy()))
                fid.append(fidelity(m.graph, s, bank, mask, seed=i))
        out["relation_importance"] = relation_importance(mt)
        f = np.array(fid)
        out["fidelity_top10"] = float(f[:, 0].mean()) if len(f) else np.nan
        out["fidelity_random10"] = float(f[:, 1].mean()) if len(f) else np.nan
    (root / "explanations.json").write_text(json.dumps(out, indent=2))
    log.info("analyses done: %s", out)
