"""Adaptive-attacker evasion suite.

Each attack changes only what an attacker controls, applied to test delegations of malicious
contracts (benign rows are untouched) and evaluated identically for every model:

  self_submit      victim's own account submits the authorization (no sponsor, no batching)
  fresh_sponsor    a brand-new submitter wallet per victim (no sponsor history / funding trail)
  fresh_contract   a freshly deployed contract per victim (no earlier delegators: no sibling
                   evidence, no delegation counters on the contract node)
  code_disguise    malicious bytecode re-deployed as a semantics-preserving mutated copy
  combined         all of the above at once
"""
from __future__ import annotations

import numpy as np

from .graph import AUTH, NODE_FEATS, SPONSOR
from .patterns.motifs import MOTIF_NAMES

SPONSOR_MOTIFS = [MOTIF_NAMES.index(n) for n in ("gas_funding", "sponsor_authorities", "sponsor_same_contract_recent",
                                                   "sponsor_contracts", "sponsor_age", "sponsor_funded_by_sink")]
SIBLING_MOTIFS = [MOTIF_NAMES.index(n) for n in MOTIF_NAMES if n.startswith("sib_")] + \
    [MOTIF_NAMES.index("sink_fanout")]
CONTRACT_HISTORY = [NODE_FEATS.index(n) for n in ("delegated_to", "new_sponsor", "in_cnt", "new_cp", "ev_1h", "ev_24h",
                                                  "age", "recency")]
ATTACKS = ["self_submit", "fresh_sponsor", "fresh_contract", "code_disguise", "combined"]


def _drop_edges(s, keep):
    return {**s, "esrc": s["esrc"][keep], "edst": s["edst"][keep], "etype": s["etype"][keep],
            "edt": s["edt"][keep], "efeat": s["efeat"][keep]}


def self_submit(s):
    """Sample-level self-submission (used to augment training data; evaluation rebuilds samples
    with the sponsor masked in the sampler)."""
    a, sp = s["centres"][0], s["centres"][1]
    if sp == a:
        return s
    keep = ~((s["etype"] == SPONSOR) & (s["edst"] == a))
    t = _drop_edges(s, keep)
    m = t["motifs"].copy()
    m[SPONSOR_MOTIFS] = 0.0
    m[MOTIF_NAMES.index("batch")] = 0.0
    f = t["authf"].copy()
    f[0], f[1] = 0.0, 1.0                     # no batch, self-sponsored
    c = t["centres"].copy()
    c[1] = a
    return {**t, "motifs": m, "authf": f, "centres": c}


AUGMENT = {"self_submit": lambda s: self_submit(s), "fresh_sponsor": lambda s: fresh_sponsor(s),
           "fresh_contract": lambda s: fresh_contract(s),
           "combined": lambda s: fresh_contract(self_submit(s))}


def fresh_sponsor(s):
    sp = s["centres"][1]
    if sp == s["centres"][0]:           # already self-submitted
        return s
    a = s["centres"][0]
    touch = (s["esrc"] == sp) | (s["edst"] == sp)
    own = (s["etype"] == SPONSOR) & (s["esrc"] == sp) & (s["edst"] == a)    # the one sponsoring edge stays
    s = _drop_edges(s, ~touch | own)
    x = s["x"].copy()
    x[sp, :-1] = 0.0
    m = s["motifs"].copy()
    m[SPONSOR_MOTIFS] = 0.0
    return {**s, "x": x, "motifs": m}


def fresh_contract(s):
    a, c = s["centres"][0], s["centres"][2]
    touch = (s["esrc"] == c) | (s["edst"] == c)
    own = (s["etype"] == AUTH) & (s["esrc"] == a) & (s["edst"] == c)        # this delegation stays
    s = _drop_edges(s, ~touch | own)
    x = s["x"].copy()
    x[c, CONTRACT_HISTORY] = 0.0
    m = s["motifs"].copy()
    m[SIBLING_MOTIFS] = 0.0
    m[MOTIF_NAMES.index("sponsor_same_contract_recent")] = 0.0
    return {**s, "x": x, "motifs": m}


def apply_attack(name, samples, masked_samples, bank, chain, strength=2.0, seed=0):
    """Attacked copy of `samples` (malicious rows only); `masked_samples` are the same rows built
    with the sponsor masked. Code disguise is applied through the bank (caller clears it)."""
    out = []
    for s, sm in zip(samples, masked_samples):
        if s["y"] != 1:
            out.append(s)
            continue
        t = s
        if name in ("self_submit", "combined"):
            t = sm
        if name in ("fresh_sponsor",):
            t = fresh_sponsor(t)
        if name in ("fresh_contract", "combined"):
            t = fresh_contract(t)
        out.append(t)
    if name in ("code_disguise", "combined"):
        bank.mutate(chain, {int(s["nodes"][s["centres"][2]]) for s in samples if s["y"] == 1}, strength, seed)
    return out
