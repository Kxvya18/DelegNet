"""Paper numbers that are not produced by the main pipeline or results/make_figures.py:

  false_alarm_bound.csv       exact pre-signature false alarms per method, with a Clopper-Pearson
                              95% upper bound on the rate
  token_exposure.csv          how much of the labelling could be affected by not observing token transfers
  significance_delegnet.csv   paired bootstrap AUROC of DelegNet (OpcodeKNN) vs each baseline on the
                              zero-day predictions, per slice, mean over seeds

  source .env.local; PYTHONPATH=. .venv/bin/python results/supporting_numbers.py
"""
from pathlib import Path

from results.paper_names import name, paper_only


def false_alarm_bound(cfg, tg):
    import pandas as pd
    from scipy.stats import beta

    from delegnet.experiments import zeroday_folds

    pre = paper_only(pd.read_csv(Path(cfg["run_root"]) / "presignature.csv"))
    rows = []
    for method, g in pre.groupby("method"):
        n = alarms = 0
        for seed, rate in g.drop_duplicates("seed").set_index("seed").false_alarm.items():
            # pre-signature screening is scored on the last zero-day fold of each seed
            te = list(zeroday_folds(tg.sort_values("event"), cfg["experiment"]["zeroday_folds"], int(seed)))[-1][3]
            nb = int((te.label == 0).sum())
            n += nb
            alarms += round(rate * nb)
        upper = beta.ppf(0.975, alarms + 1, n - alarms) if alarms < n else 1.0
        rows.append(dict(method=name(method), benign_requests=n, false_alarms=alarms,
                         rate=alarms / n, upper95_clopper_pearson=upper))
    return pd.DataFrame(rows)


def token_exposure(cfg):
    import pandas as pd

    from delegnet.patterns import code_idioms as ci

    root = Path(cfg["data_root"]) / "ethereum"
    labels = pd.read_parquet(root / "contract_labels.parquet")
    code = pd.read_parquet(root / "code.parquet")
    code = dict(zip(code.address.str.lower(), code.code))
    ix = {n: i for i, n in enumerate(ci.IDIOM_NAMES)}
    token = ["erc20_transfer", "erc20_transferFrom", "approve_or_setApprovalForAll", "permit"]
    rows = []
    for delegate, r in labels.iterrows():
        c = code.get(delegate.lower())
        if c is None:
            continue
        b = c if isinstance(c, (bytes, bytearray)) else bytes.fromhex(c.removeprefix("0x"))
        v = ci.idioms(b)
        rows.append(dict(label=int(r.label), n_auth=int(r.n_auth), drained=r.n_drained > 0,
                         token_interface=any(v[ix[k]] > 0 for k in token),
                         owner_gate=v[ix["caller_eq_hardcoded"]] > 0))
    d = pd.DataFrame(rows)
    d["drainer_profile"] = d.token_interface & d.owner_gate
    out = d.groupby("label").agg(contracts=("label", "size"), token_interface=("token_interface", "sum"),
                                 drainer_profile=("drainer_profile", "sum"),
                                 drainer_profile_no_ether_drain=("drainer_profile", lambda s: int((s & ~d.loc[s.index, "drained"]).sum())),
                                 delegations=("n_auth", "sum"),
                                 drainer_profile_delegations=("n_auth", lambda s: int(s[d.loc[s.index, "drainer_profile"]].sum())))
    return out.reset_index().replace({"label": {-1: "unlabelled", 0: "benign", 1: "malicious"}})


def significance_table(cfg):
    import pandas as pd

    from delegnet.diagnostics import significance

    pr = paper_only(pd.read_parquet(Path(cfg["run_root"]) / "zeroday" / "predictions.parquet"))
    s = significance(pr, ours="OpcodeKNN", metrics=("auroc",), n_boot=1000)
    s["vs"] = s.vs.map(name)
    return s.groupby(["slice", "vs"])[["diff", "ci_lo", "ci_hi", "p_value"]].mean().round(4).reset_index()


def main():
    import pandas as pd

    from delegnet.config import load_config

    cfg = load_config("configs/paper.yaml")
    # zero-day folds need only labels and code families, so the graph itself is not loaded
    tg = pd.read_parquet(Path(cfg["data_root"]) / cfg["experiment"]["source_chain"] / "targets.parquet",
                         columns=["auth_id", "event", "label", "family"])
    tables = Path("results/tables")
    false_alarm_bound(cfg, tg).to_csv(tables / "false_alarm_bound.csv", index=False)
    token_exposure(cfg).to_csv(tables / "token_exposure.csv", index=False)
    significance_table(cfg).to_csv(tables / "significance_delegnet.csv", index=False)


if __name__ == "__main__":
    main()
