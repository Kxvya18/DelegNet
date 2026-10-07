"""Ground-truth labels for delegations.

Two sources, recorded separately so the paper can report both:
  * curated lists (security-firm reports, paper artifacts, explorer phishing tags), and
  * behavioural outcomes: what happened to the authorities *after* they delegated.

Behavioural labels are contract-level. A contract is malicious when several of its
delegators lose funds through outflows triggered by someone else (tx.from != authority)
that converge on shared EOA sinks; benign when many delegators keep using their account
themselves and nobody is drained. Everything else stays unlabelled and is excluded from
supervised training, which is the conservative choice.

Labels only use information from after each delegation, while the detector only sees
information up to its checkpoint, so the two never overlap.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .collect import DELEGATION_PREFIX, ZERO, load_chain

log = logging.getLogger("labels")


def delegations(auth: pd.DataFrame) -> pd.DataFrame:
    """One row per authorization with the period during which it stayed in force."""
    a = auth.sort_values(["block", "tx_index", "auth_index"]).reset_index(drop=True)
    a["auth_id"] = np.arange(len(a))
    a["until_block"] = a.groupby("authority").block.shift(-1).fillna(np.iinfo(np.int64).max).astype(np.int64)
    return a


def is_eoa(code: pd.DataFrame) -> set[str]:
    c = code.dropna()
    return set(c.address[(c.code == "0x") | c.code.str.startswith(DELEGATION_PREFIX)])


def outflows_under_delegation(dl: pd.DataFrame, tr: pd.DataFrame, block_ts) -> pd.DataFrame:
    """Transfers out of each authority while its delegation was active, tagged as
    self-initiated or third-party-triggered."""
    t = tr.dropna(subset=["sender"]).merge(
        dl[~dl.revoke][["auth_id", "authority", "delegate", "block", "tx_index", "until_block"]],
        left_on="src", right_on="authority", suffixes=("", "_auth"))
    after = (t.block > t.block_auth) | ((t.block == t.block_auth) & (t.tx_index >= t.tx_index_auth))
    t = t[after & (t.block < t.until_block)].copy()
    t["third_party"] = t.sender != t.src
    t["dt_s"] = block_ts(t.block.values) - block_ts(t.block_auth.values)
    return t


def contract_evidence(dl, out, eoas, code_addrs, cfg_l) -> pd.DataFrame:
    window = cfg_l["drain_window_s"]
    # NB: the attacker is often sponsor, tx sender *and* recipient at once, so recipient == sender
    # must stay in; relayer fees are filtered by the concentration test below instead
    third = out[out.third_party & (out.dt_s <= window)]
    drains = third[third.dst.isin(eoas)]
    # sinks that collect third-party-triggered outflows from >= 2 authorities of the same contract
    per_sink = drains.groupby(["delegate", "dst"]).authority.nunique().rename("n_auth").reset_index()
    shared = per_sink[per_sink.n_auth >= 2][["delegate", "dst"]].assign(shared=True)
    drains = drains.merge(shared, on=["delegate", "dst"], how="left").fillna({"shared": False})

    selfact = out[~out.third_party].groupby("delegate").authority.nunique()
    targets = dl[~dl.revoke]
    ev = pd.DataFrame({"n_auth": targets.groupby("delegate").authority.nunique()})
    ev["n_drained"] = drains.groupby("delegate").authority.nunique()
    ev["n_shared_drained"] = drains[drains.shared].groupby("delegate").authority.nunique()
    ev["n_self_active"] = selfact
    ev = ev.fillna(0)
    ev["drained_ratio"] = ev.n_drained / ev.n_auth
    ev["shared_sink_ratio"] = ev.n_shared_drained / ev.n_drained.clip(lower=1)
    ev["self_active_ratio"] = ev.n_self_active / ev.n_auth
    # share of all third-party outflow events that land in shared EOA sinks
    ev["sink_event_share"] = (drains[drains.shared].groupby("delegate").size()
                              / third.groupby("delegate").size()).reindex(ev.index).fillna(0)
    ev["has_code"] = ev.index.isin(code_addrs)
    return ev


def label_chain(cfg: dict, chain: str) -> pd.DataFrame:
    root = Path(cfg["data_root"]) / chain
    d = load_chain(root)
    auth, tr, code = d["auth"], d["transfers"], d["code"]
    cl = cfg["labels"]
    dl = delegations(auth)
    bt = auth.groupby("block").ts.first()
    block_ts = lambda b: np.interp(b, bt.index.values, bt.values)  # noqa: E731

    # anything that ever signed a transaction is an EOA, code lookup or not
    eoas = is_eoa(code) | set(auth.sponsor) | set(auth.authority) | set(tr.sender.dropna())
    code_addrs = set(code.dropna().address[code.code.str.len() > 2])
    out = outflows_under_delegation(dl, tr, block_ts)
    ev = contract_evidence(dl, out, eoas, code_addrs, cl)
    drains = out[out.third_party & (out.dt_s <= cl["drain_window_s"]) & out.dst.isin(eoas)]

    mal = (ev.n_drained >= cl["min_drained_authorities"]) & (ev.drained_ratio >= cl["min_drained_ratio"]) & \
          (ev.shared_sink_ratio >= cl["min_shared_sink_ratio"]) & (ev.sink_event_share >= cl["min_sink_event_share"]) & \
          (ev.self_active_ratio <= cl["max_malicious_self_active"])
    ben = (ev.n_auth >= cl["min_benign_authorities"]) & (ev.drained_ratio <= cl["max_benign_drained_ratio"]) & \
          (ev.self_active_ratio >= cl["min_benign_self_active"])
    ev["label"] = np.where(mal, 1, np.where(ben, 0, -1))
    ev["source"] = np.where(ev.label >= 0, "behaviour", "none")

    for kind, val in (("benign_lists", 0), ("malicious_lists", 1)):  # malicious lists win ties
        for p in cl.get(kind, []):
            if Path(p).exists():
                lst = pd.read_csv(p)
                hit = ev.index.isin(lst.address.str.lower())
                ev.loc[hit, "label"], ev.loc[hit, "source"] = val, "list:" + Path(p).stem
    # independent cross-check: are the contract, its sponsors or its drain sinks on a public
    # scam list (e.g. ScamSniffer)? Reported as agreement, never used to train.
    scam = set()
    for p in cl.get("scam_address_lists", []):
        if Path(p).exists():
            scam |= set(pd.read_csv(p).address.str.lower())
    if scam:
        sponsors = dl[~dl.revoke].groupby("delegate").sponsor.apply(set)
        sinks = drains.groupby("delegate").dst.apply(set) if len(drains) else pd.Series(dtype=object)
        ev["scamlist_hit"] = [bool(({c} | sponsors.get(c, set()) | sinks.get(c, set())) & scam) for c in ev.index]
        for lab_val, name in ((1, "malicious"), (0, "benign")):
            sel = ev.label == lab_val
            if sel.any():
                log.info("%s: %d/%d %s contracts linked to a public scam list", chain,
                         ev.scamlist_hit[sel].sum(), sel.sum(), name)
    ev.index.name = "delegate"
    ev.to_parquet(root / "contract_labels.parquet")

    # delegation-level table: one target per non-revoking authorization
    first_drain = out[out.third_party].groupby("auth_id").agg(
        first_drain_block=("block", "min"), first_drain_tx=("tx_index", "min"))
    loss = out[out.third_party].groupby("auth_id").value.sum().rename("drained_value")
    lab = dl.merge(ev[["label", "source"]], left_on="delegate", right_index=True, how="left") \
        .join(first_drain, on="auth_id").join(loss, on="auth_id")
    lab = lab[~lab.revoke & (lab.delegate != ZERO)]
    lab["label"] = lab.label.fillna(-1).astype(int)
    lab.to_parquet(root / "delegations.parquet", index=False)

    # audit sheet: the most consequential contracts, for manual verification in the paper
    ev.sort_values("n_auth", ascending=False).head(300).to_csv(root / "label_audit.csv")
    log.info("%s: %d delegations | malicious %d | benign %d | unlabelled %d | contracts %d/%d/%d",
             chain, len(lab), (lab.label == 1).sum(), (lab.label == 0).sum(), (lab.label == -1).sum(),
             (ev.label == 1).sum(), (ev.label == 0).sum(), (ev.label == -1).sum())
    return lab
