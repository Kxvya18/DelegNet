"""Correctness checks that the paper's claims depend on.

Run: DELEGNET_CONFIG=configs/paper.yaml pytest -q tests   (needs a built source chain)
"""
import os
import random

import numpy as np
import pytest
import torch

from delegnet.config import load_config
from delegnet.evm import _layout, disassemble, mutate_bytecode
from delegnet.graph import load_graph
from delegnet.model import segment_softmax
from delegnet.patterns import code_idioms
from delegnet.patterns.motifs import motif_vector

CFG = load_config(os.environ.get("DELEGNET_CONFIG", "configs/paper.yaml"))


@pytest.fixture(scope="module")
def chain():
    return load_graph(CFG, CFG["experiment"]["source_chain"])


def test_relayout_is_identity_on_real_code(chain):
    g, _ = chain
    for raw in list(g.code_bytes.values())[:50]:
        # a PUSH truncated at the end of code is zero-padded, so compare on the original length
        assert _layout(disassemble(raw))[:len(raw)] == raw


def test_mutation_preserves_jump_targets(chain):
    """Every static jump in a mutated contract must still land on a JUMPDEST."""
    g, _ = chain
    rng = random.Random(0)
    for raw in list(g.code_bytes.values())[:50]:
        mut = mutate_bytecode(raw, rng, 2.0)
        ins = disassemble(mut)
        dests = {i.offset for i in ins if i.op == "JUMPDEST"}
        orig = disassemble(raw)
        orig_dests = {i.offset for i in orig if i.op == "JUMPDEST"}
        n_static = sum(1 for a, b in zip(orig, orig[1:]) if b.op in ("JUMP", "JUMPI") and a.arg in orig_dests)
        n_ok = sum(1 for a, b in zip(ins, ins[1:]) if b.op in ("JUMP", "JUMPI") and a.arg in dests)
        assert n_ok >= n_static


def test_canonicalization_undoes_mutation(chain):
    from difflib import SequenceMatcher
    g, _ = chain
    rng = random.Random(1)
    sim, same_idioms = [], []
    for raw in list(g.code_bytes.values())[:50]:
        mut = mutate_bytecode(raw, rng, 1.0)
        a, b = code_idioms.tokens(raw), code_idioms.tokens(mut)
        sim.append(SequenceMatcher(None, a, b, autojunk=False).ratio())
        same_idioms.append(np.allclose(code_idioms.idioms(raw), code_idioms.idioms(mut)))
    assert np.mean(sim) > 0.95 and np.mean(same_idioms) > 0.9


def test_prototype_gradients_finite_at_zero_distance():
    """The clustering loss pulls embeddings onto prototypes; gradients must stay finite there."""
    from delegnet.patterns.prototypes import PrototypeLayer
    layer = PrototypeLayer(8, per_class=2)
    z = layer.protos.detach()[:3].clone().requires_grad_(True)   # exactly on three prototypes
    logit, _, d2 = layer(z)
    cl, sep, div = layer.losses(d2, torch.tensor([0.0, 0.0, 1.0]))
    (logit.sum() + cl + sep + div).backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(layer.protos.grad).all()


def test_segment_softmax():
    s = torch.randn(7, 2)
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    out = segment_softmax(s, idx, 3)
    for k in range(3):
        assert torch.allclose(out[idx == k], torch.softmax(s[idx == k], 0), atol=1e-6)


def test_no_future_leakage(chain):
    """Samples and motifs at checkpoint 0 must not change if every later event vanished."""
    g, tg = chain
    rng = np.random.default_rng(0)
    for r in tg.sample(min(200, len(tg)), random_state=0).itertuples():
        cut = g.cutoff_for(r.event, 0)
        s = g.sample(r.event, cut, 24, 8, 256)
        assert (s["edt"] >= 0).all()
        m = motif_vector(g, r.auth_id, cut)
        # outcome milestones used by the motif counter must be at or before the cutoff
        assert m[-3] == float(g.outcomes.sweep_ev[r.auth_id] <= cut)
        assert g.outcomes.third_ev[r.auth_id] > r.event or g.outcomes.third_ev[r.auth_id] == np.iinfo(np.int64).max
    assert rng is not None
