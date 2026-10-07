"""Temporal attack-motif recognition.

EIP-7702 attacks leave recognisable typed temporal motifs:
  sweeper   gas funding -> sponsored (often batched) AUTH -> inflow -> third-party outflow
            to a shared sink within seconds
  drainer   AUTH -> short delay -> third-party-triggered outflows of every asset -> sink fan-out
and benign delegations leave their own: the owner keeps transacting, nobody else moves funds.

For every delegation we precompute *event ids* of its outcome milestones (first
third-party outflow, first sweep, first self-initiated outflow, end of delegation). A
motif count at cutoff `c` then only counts milestones with id <= c, which makes the
counter exact, fast, and leak-free even for events in the same block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AUTH, SPONSOR, REVOKE, TRANSFER = 0, 1, 2, 3
NONE = np.iinfo(np.int64).max
SWEEP_S = 60.0
DRAIN_S = 86400.0

MOTIF_NAMES = [
    "batch", "gas_funding", "sponsor_authorities", "sponsor_same_contract_recent", "sponsor_contracts",
    "sponsor_age", "sponsor_funded_by_sink",
    "sib_count", "sib_drained_frac", "sib_fast_drain_frac", "sib_sweep_frac", "sib_self_active_frac",
    "sib_left_frac", "sib_shared_sink", "sib_sponsor_diversity", "sink_fanout",
    "auth_third_out", "auth_third_val", "auth_sweep", "auth_self_out", "auth_prior_drained",
]


@dataclass
class DelegationOutcomes:
    event: np.ndarray        # auth_id -> AUTH event id
    end_ev: np.ndarray       # first later AUTH/REVOKE by the same authority
    third_ev: np.ndarray     # first outflow triggered by someone else
    first_sink: np.ndarray   # recipient of that outflow
    sweep_ev: np.ndarray     # first third-party outflow within SWEEP_S of an inflow
    self_ev: np.ndarray      # first outflow the authority triggered itself
    sib_ptr: np.ndarray      # contract node -> slice into sib_ids
    sib_ids: np.ndarray      # auth ids grouped by contract, in event order
    ev2auth: np.ndarray      # event id -> auth id (-1 if not an AUTH event)

    @classmethod
    def compute(cls, g) -> "DelegationOutcomes":
        n_auth = len(g.auth_event)
        end_ev, third_ev, sweep_ev, self_ev = (np.full(n_auth, NONE, dtype=np.int64) for _ in range(4))
        first_sink = np.full(n_auth, -1, dtype=np.int64)
        ev_to_auth = {int(e): i for i, e in enumerate(g.auth_event) if e >= 0}  # noqa: C416

        for a in np.unique(g.src[g.auth_event[g.auth_event >= 0]]):
            evs = g.inc_ev[g.indptr[a]:g.indptr[a + 1]]
            et = g.etype[evs]
            is_mine = (g.src[evs] == a) & ((et == AUTH) | (et == REVOKE))
            marks = evs[is_mine]
            starts = evs[(et == AUTH) & (g.src[evs] == a)]
            for e in starts:
                i = ev_to_auth[int(e)]
                nxt = marks[marks > e]
                end = nxt[0] if len(nxt) else NONE
                end_ev[i] = end
                w = evs[(evs > e) & (evs < end)]
                if not len(w):
                    continue
                tr = w[g.etype[w] == TRANSFER]
                out = tr[g.src[tr] == a]
                third = out[g.efeat[out, 3] > 0]
                own = out[g.efeat[out, 3] == 0]
                if len(third):
                    third_ev[i], first_sink[i] = third[0], g.dst[third[0]]
                    inflow_t = g.ts[tr[g.dst[tr] == a]]
                    if len(inflow_t):
                        k = np.searchsorted(inflow_t, g.ts[third], side="right") - 1
                        ok = (k >= 0) & (g.ts[third] - inflow_t[np.maximum(k, 0)] <= SWEEP_S)
                        if ok.any():
                            sweep_ev[i] = third[ok][0]
                if len(own):
                    self_ev[i] = own[0]

        # siblings: auth ids per contract, in event order
        valid = np.flatnonzero(g.auth_event >= 0)
        contract = g.dst[g.auth_event[valid]]
        order = np.lexsort((g.auth_event[valid], contract))
        sib_ids = valid[order]
        sib_ptr = np.zeros(len(g.addr) + 1, dtype=np.int64)
        np.add.at(sib_ptr, contract + 1, 1)
        ev2auth = np.full(len(g.src), -1, dtype=np.int64)
        ev2auth[g.auth_event[valid]] = valid
        return cls(g.auth_event, end_ev, third_ev, first_sink, sweep_ev, self_ev, np.cumsum(sib_ptr), sib_ids,
                   ev2auth)

    def siblings(self, contract: int, cutoff: int, exclude: int, limit: int) -> np.ndarray:
        ids = self.sib_ids[self.sib_ptr[contract]:self.sib_ptr[contract + 1]]
        k = np.searchsorted(self.event[ids], cutoff, side="right")
        ids = ids[:k]
        ids = ids[ids != exclude]
        return ids, ids[-limit:]


def motif_vector(g, auth_id: int, cutoff: int, max_siblings: int = 256, delta: float = 3600.0,
                 mask_sponsor: bool = False) -> np.ndarray:
    o = g.outcomes
    e = o.event[auth_id]
    a, c = g.src[e], g.dst[e]
    s = a if mask_sponsor else g.sender[e]
    t0, tc = g.ts[e], g.ts[cutoff]
    f = dict.fromkeys(MOTIF_NAMES, 0.0)
    f["batch"] = 0.0 if mask_sponsor else g.efeat[e, 4]

    a_ev = g.node_events(a, cutoff)
    tr_in = a_ev[(g.etype[a_ev] == TRANSFER) & (g.dst[a_ev] == a)]

    # --- sponsor / campaign infrastructure
    if s != a:
        f["gas_funding"] = np.log1p(((g.src[tr_in] == s) & (g.ts[tr_in] >= t0 - delta)).sum())
        s_ev = g.node_events(s, cutoff)
        sp = s_ev[(g.etype[s_ev] == SPONSOR) & (g.src[s_ev] == s)]
        f["sponsor_authorities"] = np.log1p(len(np.unique(g.dst[sp])))
        f["sponsor_same_contract_recent"] = np.log1p(((g.contract_of[sp] == c) & (g.ts[sp] >= tc - delta)).sum())
        f["sponsor_contracts"] = np.log1p(len(np.unique(g.contract_of[sp])))
        f["sponsor_age"] = np.log1p(tc - g.ts[s_ev[0]]) if len(s_ev) else 0.0
        s_in = s_ev[(g.etype[s_ev] == TRANSFER) & (g.dst[s_ev] == s)]
        funders = set(g.src[s_in].tolist())
    else:
        funders = set()

    # --- what already happened to earlier delegators of the same contract
    all_sib, sib = o.siblings(c, cutoff, auth_id, max_siblings)
    if len(sib):
        f["sib_count"] = np.log1p(len(all_sib))
        drained = (o.third_ev[sib] <= cutoff) & (g.ts[np.minimum(o.third_ev[sib], len(g.ts) - 1)]
                                                  - g.ts[o.event[sib]] <= DRAIN_S)
        third_t = g.ts[np.minimum(o.third_ev[sib], len(g.ts) - 1)]
        f["sib_drained_frac"] = drained.mean()
        f["sib_fast_drain_frac"] = (drained & (third_t - g.ts[o.event[sib]] <= SWEEP_S)).mean()
        f["sib_sweep_frac"] = (o.sweep_ev[sib] <= cutoff).mean()
        f["sib_self_active_frac"] = (o.self_ev[sib] <= cutoff).mean()
        f["sib_left_frac"] = (o.end_ev[sib] <= cutoff).mean()
        f["sib_sponsor_diversity"] = len(np.unique(g.sender[o.event[sib]])) / len(sib)
        sinks = o.first_sink[sib][drained]
        if len(sinks):
            vals, counts = np.unique(sinks, return_counts=True)
            top = vals[np.argmax(counts)]
            f["sib_shared_sink"] = np.log1p(counts.max())
            k_ev = g.node_events(top, cutoff)
            out = k_ev[(g.etype[k_ev] == TRANSFER) & (g.src[k_ev] == top) & (g.ts[k_ev] >= tc - 24 * delta)]
            f["sink_fanout"] = np.log1p(len(np.unique(g.dst[out])))
            f["sponsor_funded_by_sink"] = float(bool(funders & set(vals.tolist())))

    # --- the authority itself, after delegating (only visible at later checkpoints)
    after = a_ev[(a_ev > e) & (g.etype[a_ev] == TRANSFER) & (g.src[a_ev] == a)]
    third = after[g.efeat[after, 3] > 0]
    f["auth_third_out"] = np.log1p(len(third))
    f["auth_third_val"] = np.log1p(np.expm1(g.efeat[third, 0]).sum()) if len(third) else 0.0
    f["auth_sweep"] = float(o.sweep_ev[auth_id] <= cutoff)
    f["auth_self_out"] = np.log1p(len(after) - len(third))
    mine = a_ev[(g.etype[a_ev] == AUTH) & (g.src[a_ev] == a) & (a_ev < e)]
    if len(mine):
        prior = o.ev2auth[mine]
        f["auth_prior_drained"] = float((o.third_ev[prior] <= cutoff).any())
    return np.array([f[k] for k in MOTIF_NAMES], dtype=np.float32)
