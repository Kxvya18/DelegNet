"""Heterogeneous temporal graph over delegations and value transfers.

Nodes are addresses (EOA or contract). Events, in chain order:
  AUTH      authority -> delegate contract
  SPONSOR   tx sender -> authority (only when someone else submitted the authorization)
  REVOKE    authority -> contract it is leaving (re-delegation or reset to 0x0)
  TRANSFER  src -> dst, native or ERC-20, with who triggered it

Everything the model or the motif counter sees for a prediction is restricted to events
with id <= cutoff, so a checkpoint never peeks past its own time.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .collect import ZERO, load_chain
from .labels import delegations
from .patterns import code_idioms
from .patterns.motifs import MOTIF_NAMES, DelegationOutcomes, motif_vector

log = logging.getLogger("graph")
AUTH, SPONSOR, REVOKE, TRANSFER = 0, 1, 2, 3
N_REL = 4
EDGE_FEATS = ["value", "is_token", "internal", "third_party", "log_batch", "self_sponsored"]
CUM_FEATS = ["out_cnt", "in_cnt", "out_val", "in_val", "new_cp", "auth_signed", "auth_sponsored",
             "delegated_to", "third_party_out", "revokes", "token_out", "internal_in", "new_sponsor"]
NODE_FEATS = CUM_FEATS + ["age", "recency", "ev_1h", "ev_24h", "is_contract"]
AUTH_FEATS = ["log_batch", "self_sponsored", "first_delegation", "log_checkpoint", "redelegation"]


@dataclass
class TemporalGraph:
    chain: str
    addr: np.ndarray
    is_contract: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    etype: np.ndarray
    ts: np.ndarray
    sender: np.ndarray
    contract_of: np.ndarray          # for AUTH/SPONSOR events: the delegate contract, else -1
    efeat: np.ndarray
    indptr: np.ndarray
    inc_ev: np.ndarray
    inc_cum: np.ndarray
    auth_event: np.ndarray           # auth_id -> event id
    outcomes: DelegationOutcomes = None
    code_tokens: dict = field(default_factory=dict)   # node -> token id array
    code_idioms: dict = field(default_factory=dict)   # node -> idiom vector
    code_bytes: dict = field(default_factory=dict)    # node -> raw bytecode

    # ---------------------------------------------------------------- queries
    def node_events(self, n: int, cutoff: int, last: int | None = None) -> np.ndarray:
        s, e = self.indptr[n], self.indptr[n + 1]
        k = s + np.searchsorted(self.inc_ev[s:e], cutoff, side="right")
        return self.inc_ev[max(s, k - last) if last else s:k]

    def node_stats(self, nodes: np.ndarray, cutoff: int) -> np.ndarray:
        t_now = self.ts[cutoff]
        out = np.zeros((len(nodes), len(NODE_FEATS)), dtype=np.float32)
        for r, n in enumerate(nodes):
            s, e = self.indptr[n], self.indptr[n + 1]
            k = np.searchsorted(self.inc_ev[s:e], cutoff, side="right")
            if k:
                out[r, :len(CUM_FEATS)] = self.inc_cum[s + k - 1]
                evts = self.ts[self.inc_ev[s:s + k]]
                out[r, len(CUM_FEATS)] = t_now - evts[0]
                out[r, len(CUM_FEATS) + 1] = t_now - evts[-1]
                out[r, len(CUM_FEATS) + 2] = k - np.searchsorted(evts, t_now - 3600)
                out[r, len(CUM_FEATS) + 3] = k - np.searchsorted(evts, t_now - 86400)
            out[r, -1] = self.is_contract[n]
        out[:, :-1] = np.log1p(np.maximum(out[:, :-1], 0))
        return out

    def cutoff_for(self, auth_ev: int, delta_s: float) -> int:
        """Last event id visible `delta_s` seconds after the authorization; delta_s < 0 means the
        moment of signing, before the authorization itself is on chain."""
        if delta_s < 0:
            return auth_ev - 1
        if delta_s == 0:
            return auth_ev
        return int(np.searchsorted(self.ts, self.ts[auth_ev] + delta_s, side="right") - 1)

    # ---------------------------------------------------------------- sampling
    def sample(self, auth_ev: int, cutoff: int, k1: int, k2: int, max_nodes: int,
               mask_sponsor: bool = False) -> dict:
        a, c = self.src[auth_ev], self.dst[auth_ev]
        s = self.sender[auth_ev]
        if mask_sponsor:   # structural-evasion test: pretend the victim submitted it
            s = a
        centres = [a, s, c]
        evs = [self.node_events(n, cutoff, k1) for n in dict.fromkeys(centres)]
        hop1 = np.unique(np.concatenate(evs))
        nbrs = pd.unique(np.concatenate([self.src[hop1], self.dst[hop1]]))
        nbrs = [n for n in nbrs if n not in centres][: max_nodes // 2]
        hop2 = [self.node_events(n, cutoff, k2) for n in nbrs]
        ev = np.unique(np.concatenate([hop1, *hop2])) if hop2 else hop1
        if mask_sponsor:
            ev = ev[~((self.etype[ev] == SPONSOR) & (self.dst[ev] == a))]

        nodes = pd.unique(np.concatenate([centres, self.src[ev], self.dst[ev]]))
        if len(nodes) > max_nodes:   # keep centres, drop the oldest peripheral events
            keep = set(nodes[:max_nodes])
            m = np.array([self.src[x] in keep and self.dst[x] in keep for x in ev], dtype=bool)
            ev, nodes = ev[m], nodes[:max_nodes]
        local = {n: i for i, n in enumerate(nodes)}
        role = np.full(len(nodes), 3, dtype=np.int64)
        for r, n in enumerate(centres):
            role[local[n]] = min(role[local[n]], r)

        efeat = self.efeat[ev].copy()
        if mask_sponsor:
            m = ev == auth_ev
            efeat[m, EDGE_FEATS.index("self_sponsored")] = 1.0
        return dict(
            nodes=np.asarray(nodes, dtype=np.int64),
            x=self.node_stats(np.asarray(nodes), cutoff),
            role=role,
            is_contract=self.is_contract[nodes].astype(np.int64),
            esrc=np.array([local[x] for x in self.src[ev]], dtype=np.int64),
            edst=np.array([local[x] for x in self.dst[ev]], dtype=np.int64),
            etype=self.etype[ev].astype(np.int64),
            edt=(self.ts[cutoff] - self.ts[ev]).astype(np.float32),
            efeat=efeat,
            centres=np.array([local[a], local[s], local[c]], dtype=np.int64),
        )

    def auth_features(self, auth_ev: int, cutoff: int, mask_sponsor=False) -> np.ndarray:
        a = self.src[auth_ev]
        prev = self.node_events(a, auth_ev - 1)
        prev_auth = (self.etype[prev] == AUTH).sum() if len(prev) else 0
        f = self.efeat[auth_ev]
        return np.array([0.0 if mask_sponsor else f[4], 1.0 if mask_sponsor else f[5],
                         float(prev_auth == 0), np.log1p(max(self.ts[cutoff] - self.ts[auth_ev], 0.0)),
                         float(prev_auth > 0)], dtype=np.float32)


# -------------------------------------------------------------------- building

def _cumulative(src, dst, etype, efeat, sender, contract_of, n_nodes):
    """CSR incidence (node -> its events in order) plus running per-node counters."""
    E = len(src)
    ev = np.arange(E)
    node = np.concatenate([src, dst])
    evx = np.concatenate([ev, ev])
    is_src = np.concatenate([np.ones(E, bool), np.zeros(E, bool)])
    self_loop = np.concatenate([np.zeros(E, bool), src == dst])
    node, evx, is_src = node[~self_loop], evx[~self_loop], is_src[~self_loop]
    order = np.lexsort((evx, node))
    node, evx, is_src = node[order], evx[order], is_src[order]
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(indptr, node + 1, 1)
    indptr = np.cumsum(indptr)

    t = etype[evx]
    val = efeat[evx, 0]
    other = np.where(is_src, dst[evx], src[evx])
    is_tr = t == TRANSFER
    flags = np.zeros((len(node), len(CUM_FEATS)), dtype=np.float32)
    flags[:, 0] = is_tr & is_src
    flags[:, 1] = is_tr & ~is_src
    flags[:, 2] = np.where(is_tr & is_src, val, 0)
    flags[:, 3] = np.where(is_tr & ~is_src, val, 0)
    flags[:, 4] = ~pd.DataFrame({"n": node, "o": other}).duplicated().values
    flags[:, 5] = (t == AUTH) & is_src
    flags[:, 6] = (t == SPONSOR) & is_src
    flags[:, 7] = (t == AUTH) & ~is_src
    flags[:, 8] = is_tr & is_src & (efeat[evx, 3] > 0)
    flags[:, 9] = t == REVOKE
    flags[:, 10] = is_tr & is_src & (efeat[evx, 1] > 0)
    flags[:, 11] = is_tr & ~is_src & (efeat[evx, 2] > 0)
    spons = np.where((t == AUTH) & ~is_src, sender[evx], -1)
    flags[:, 12] = (spons >= 0) & ~pd.DataFrame({"n": node, "s": spons}).duplicated().values

    cum = np.cumsum(flags, axis=0, dtype=np.float64)
    seg_start = np.repeat(indptr[:-1], np.diff(indptr))
    base = np.where(seg_start > 0, cum[np.maximum(seg_start - 1, 0)].T, 0).T
    return indptr, evx.astype(np.int64), (cum - base).astype(np.float32)


def build_graph(chain: str, auth: pd.DataFrame, tr: pd.DataFrame, code: pd.DataFrame,
                norm_frac: float = 0.2) -> tuple[TemporalGraph, pd.DataFrame]:
    dl = delegations(auth)
    bt = auth.groupby("block").ts.first()
    block_ts = lambda b: np.interp(b, bt.index.values, bt.values)  # noqa: E731

    # value normalisation per asset with medians from the earliest part of the window only
    tr = tr.copy()
    t_cut = bt.index.values[0] + norm_frac * (bt.index.values[-1] - bt.index.values[0])
    med = tr[tr.block <= t_cut].groupby("asset").value.median()
    tr["vnorm"] = np.log1p(tr.value / tr.asset.map(med).fillna(tr.groupby("asset").value.transform("median")))

    # previous delegate of each authority, for REVOKE edges
    dl["prev_delegate"] = dl.groupby("authority").delegate.shift(1)

    rows = []
    for r in dl.itertuples():
        base = dict(block=r.block, tx_index=r.tx_index, sender=r.sponsor, contract=r.delegate,
                    auth_id=r.auth_id, value=0.0, is_token=0, internal=0, third_party=0,
                    log_batch=np.log1p(r.batch_size), self_sponsored=float(r.sponsor == r.authority))
        if isinstance(r.prev_delegate, str) and r.prev_delegate != ZERO:
            rows.append({**base, "sub": -3000 + r.auth_index, "etype": REVOKE, "src": r.authority,
                         "dst": r.prev_delegate, "contract": r.prev_delegate})
        if r.sponsor != r.authority:
            rows.append({**base, "sub": -2000 + r.auth_index, "etype": SPONSOR, "src": r.sponsor,
                         "dst": r.authority})
        if r.delegate != ZERO:
            rows.append({**base, "sub": -1000 + r.auth_index, "etype": AUTH, "src": r.authority,
                         "dst": r.delegate})
    ev_auth = pd.DataFrame(rows)
    ev_auth.loc[ev_auth.etype != AUTH, "auth_id"] = -1

    ev_tr = pd.DataFrame(dict(
        block=tr.block, tx_index=tr.tx_index, sub=tr["sub"], etype=TRANSFER, src=tr.src, dst=tr.dst,
        sender=tr.sender.fillna(""), contract=None, auth_id=-1, value=tr.vnorm,
        is_token=(tr.asset != "native").astype(float), internal=tr.internal.astype(float),
        third_party=(tr.sender.notna() & (tr.sender != tr.src)).astype(float), log_batch=0.0,
        self_sponsored=0.0))
    ev = pd.concat([ev_auth, ev_tr], ignore_index=True).sort_values(
        ["block", "tx_index", "sub"], kind="stable").reset_index(drop=True)

    addrs = pd.unique(pd.concat([ev.src, ev.dst, ev.sender[ev.sender != ""]]))
    idx = pd.Index(addrs)
    codes = code.dropna().set_index("address").code
    is_contract = np.array([isinstance(codes.get(a), str) and len(codes.get(a)) > 2
                            and not codes.get(a).startswith("0xef0100") for a in addrs])
    src, dst = idx.get_indexer(ev.src), idx.get_indexer(ev.dst)
    sender = np.where(ev.sender != "", idx.get_indexer(ev.sender.replace("", addrs[0])), -1)
    contract_of = np.where(ev.contract.notna(), idx.get_indexer(ev.contract.fillna(addrs[0])), -1)
    efeat = ev[EDGE_FEATS].to_numpy(np.float32)
    etype = ev.etype.to_numpy(np.int8)
    ts = block_ts(ev.block.values).astype(np.float64)
    ts = np.maximum.accumulate(ts)
    indptr, inc_ev, inc_cum = _cumulative(src, dst, etype, efeat, sender, contract_of, len(addrs))

    auth_event = np.full(int(dl.auth_id.max()) + 1, -1, dtype=np.int64)
    m = ev.etype == AUTH
    auth_event[ev.auth_id[m].astype(int).values] = np.flatnonzero(m.values)

    g = TemporalGraph(chain, np.asarray(addrs), is_contract, src, dst, etype, ts, sender, contract_of,
                      efeat, indptr, inc_ev, inc_cum, auth_event)
    g.outcomes = DelegationOutcomes.compute(g)

    for a, hexcode in codes.items():
        n = idx.get_indexer([a])[0]
        if n >= 0 and is_contract[n]:
            raw = bytes.fromhex(hexcode[2:])
            g.code_bytes[n] = raw
            g.code_tokens[n] = encode_tokens(code_idioms.tokens(raw))
            g.code_idioms[n] = code_idioms.idioms(raw)
    return g, dl


# -------------------------------------------------------------------- code vocabulary

from .evm import OPCODE_NAMES  # noqa: E402

VOCAB = ["<pad>", "<cls>", "<unk>"] + OPCODE_NAMES + sorted(set(code_idioms.SELECTORS.values())) + \
        ["ADDR20", "PUSHTAG"] + [f"PUSH{w}_SMALL" for w in range(1, 33)]
TOKEN_ID = {t: i for i, t in enumerate(VOCAB)}


def encode_tokens(toks: list[str], max_len: int = 512) -> np.ndarray:
    ids = [1] + [TOKEN_ID.get(t, 2) for t in toks[: max_len - 1]]
    return np.array(ids, dtype=np.int64)


# -------------------------------------------------------------------- per-chain build

def build_chain(cfg: dict, chain: str):
    root = Path(cfg["data_root"]) / chain
    d = load_chain(root)
    g, _ = build_graph(chain, d["auth"], d["transfers"], d["code"])
    lab = pd.read_parquet(root / "delegations.parquet")
    lab["event"] = g.auth_event[lab.auth_id.values]
    lab = lab[lab.event >= 0].copy()
    # one target per (authority, contract): sweepers re-authorize the same victim many times, and
    # near-duplicate targets would leak between train and test
    lab = lab.sort_values("event").drop_duplicates(["authority", "delegate"], keep="first")
    lab["contract_node"] = g.dst[lab.event.values]
    lab["family"] = code_families(g, lab.contract_node.values)
    with open(root / "graph.pkl", "wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
    lab.to_parquet(root / "targets.parquet", index=False)
    log.info("%s: %d nodes, %d events, %d targets, %d code families", chain, len(g.addr), len(g.src),
             len(lab), lab.family.nunique())


def load_graph(cfg: dict, chain: str) -> tuple[TemporalGraph, pd.DataFrame]:
    root = Path(cfg["data_root"]) / chain
    with open(root / "graph.pkl", "rb") as f:
        g = pickle.load(f)
    return g, pd.read_parquet(root / "targets.parquet")


def code_families(g: TemporalGraph, contract_nodes: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """Group contracts whose canonical opcode 4-gram sets overlap (Jaccard >= threshold).
    These families drive the zero-day (leave-family-out) and family-disjoint splits."""
    uniq = np.unique(contract_nodes)
    grams = {}
    for n in uniq:
        toks = g.code_tokens.get(n, np.array([2]))
        grams[n] = set(zip(toks, toks[1:], toks[2:], toks[3:]))
    parent = {n: n for n in uniq}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # bucket by exact canonical token stream first (most copies are exact), then compare reps
    by_stream = {}
    for n in uniq:
        by_stream.setdefault(g.code_tokens.get(n, np.array([2])).tobytes(), []).append(n)
    reps = []
    for members in by_stream.values():
        for m in members[1:]:
            parent[find(m)] = find(members[0])
        reps.append(members[0])
    for i, a in enumerate(reps):
        for b in reps[i + 1:]:
            ga, gb = grams[a], grams[b]
            if ga and gb and len(ga & gb) / len(ga | gb) >= threshold:
                parent[find(b)] = find(a)
    fam = {n: find(n) for n in uniq}
    return np.array([fam[n] for n in contract_nodes])


__all__ = ["TemporalGraph", "build_graph", "build_chain", "load_graph", "MOTIF_NAMES", "motif_vector",
           "NODE_FEATS", "EDGE_FEATS", "AUTH_FEATS", "N_REL", "VOCAB"]
