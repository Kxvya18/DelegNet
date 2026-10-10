from __future__ import annotations

import copy
import dataclasses
import hashlib
import logging
import multiprocessing as mp
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from .evm import mutate_bytecode
from .graph import encode_tokens, load_graph
from .model import VARIANTS, TemporalGNN
from .patterns import code_idioms
from .patterns.motifs import motif_vector


log = logging.getLogger("train")
_G = None  # graph shared with forked workers


# ------------------------------------------------------------------ samples

def _make(args):
    auth_id, event, delta, label, gcfg, mask = args
    g = _G
    cutoff = g.cutoff_for(event, delta)
    mask = mask or delta < 0          # at signing time the submitter is not known yet
    s = g.sample(event, cutoff, gcfg["k1"], gcfg["k2"], gcfg["max_nodes"], mask_sponsor=mask)
    s["motifs"] = motif_vector(g, auth_id, cutoff, gcfg["max_siblings"], gcfg["motif_delta_s"], mask_sponsor=mask)
    s["authf"] = g.auth_features(event, cutoff, mask_sponsor=mask)
    s.update(y=float(label), auth_id=auth_id, delta=delta, cutoff=cutoff, chain=g.chain)
    return s


def build_samples(cfg, chain: str, targets: pd.DataFrame, deltas, mask_sponsor=False, workers=8) -> list[dict]:
    """Samples for every (delegation, checkpoint); cached on disk by content hash."""
    global _G
    key = hashlib.md5(pickle.dumps((sorted(targets.auth_id.tolist()), list(deltas), mask_sponsor,
                                    cfg["graph"]))).hexdigest()[:16]
    f = Path(cfg["data_root"]) / chain / "samples" / f"{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    _G, _ = load_graph(cfg, chain)
    jobs = [(int(r.auth_id), int(r.event), float(d), int(r.label), cfg["graph"], mask_sponsor)
            for r in targets.itertuples() for d in deltas]
    with mp.get_context("fork").Pool(workers) as pool:
        out = pool.map(_make, jobs, chunksize=64)
    f.parent.mkdir(exist_ok=True)
    f.write_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    log.info("%s: built %d samples (%d targets x %d checkpoints)", chain, len(out), len(targets), len(deltas))
    return out


class CodeBank:
    """Token/idiom lookup for contract nodes, with optional per-contract overrides
    (mutated bytecode for the robustness experiments)."""

    def __init__(self, graphs: dict):
        self.graphs = graphs
        self.override = {}
        self.raw_override = {}

    def get(self, chain, node):
        if (chain, node) in self.override:
            return self.override[(chain, node)]
        g = self.graphs[chain]
        if node not in g.code_tokens:
            return None
        return g.code_tokens[node], g.code_idioms[node]

    def mutate(self, chain, nodes, strength, seed):
        rng = random.Random(seed)
        g = self.graphs[chain]
        for n in nodes:
            if n in g.code_bytes:
                raw = mutate_bytecode(g.code_bytes[n], rng, strength)
                self.override[(chain, n)] = (encode_tokens(code_idioms.tokens(raw)), code_idioms.idioms(raw))
                self.raw_override[(chain, n)] = raw

    def raw(self, chain, node):
        """Deployed bytecode as an attacker would ship it (mutated copy if one is active)."""
        return self.raw_override.get((chain, node), self.graphs[chain].code_bytes.get(node))


def collate(samples: list[dict], bank: CodeBank, device="cpu") -> dict:
    xs, roles, kinds = [], [], []
    es, ed, et, edt, ef, cen = [], [], [], [], [], []
    off = 0
    for s in samples:
        xs.append(s["x"]); roles.append(s["role"]); kinds.append(s["is_contract"])
        es.append(s["esrc"] + off); ed.append(s["edst"] + off); et.append(s["etype"])
        edt.append(s["edt"]); ef.append(s["efeat"]); cen.append(s["centres"] + off)
        off += len(s["nodes"])
    T = lambda a, dt=torch.float32: torch.as_tensor(np.concatenate(a), dtype=dt, device=device)  # noqa: E731
    return dict(
        x=T(xs), role=T(roles, torch.long), is_contract=T(kinds, torch.long),
        esrc=T(es, torch.long), edst=T(ed, torch.long), etype=T(et, torch.long), edt=T(edt), efeat=T(ef),
        centres=torch.as_tensor(np.stack(cen), device=device),
        authf=torch.as_tensor(np.stack([s["authf"] for s in samples]), device=device),
        y=torch.as_tensor([s["y"] for s in samples], dtype=torch.float32, device=device),
    )


# ------------------------------------------------------------------ training

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def _forward_safe(model, batch, bank, device):
    """Forward pass that survives macOS aborting GPU work ("Impacting Interactivity"):
    retry on the GPU, then fall back to the CPU for this batch. Outputs are checked for NaN."""
    for attempt in range(3):
        try:
            out = model(collate(batch, bank, device))
            if torch.isfinite(out["logit"]).all():
                return {k: v.cpu() for k, v in out.items()}
            raise RuntimeError("non-finite output")
        except (RuntimeError, getattr(torch, "AcceleratorError", RuntimeError)) as e:
            log.warning("predict batch failed on %s (%s), retry %d", device, str(e)[:60], attempt)
    model.to("cpu")
    out = model(collate(batch, bank, "cpu"))
    model.to(device)
    return {k: v.cpu() for k, v in out.items()}


@torch.no_grad()
def predict(model, samples, bank, bs=64, device="cpu") -> dict:
    model.eval()
    p, z = [], []
    for i in range(0, len(samples), bs):
        out = _forward_safe(model, samples[i:i + bs], bank, device)
        p.append(torch.sigmoid(out["logit"])); z.append(out["z"])
    return dict(p=torch.cat(p).numpy(), z=torch.cat(z).numpy())


def _loss(out, y):
    pos = (y == 0).sum().clamp_min(1) / (y == 1).sum().clamp_min(1)
    return F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos)


def train_model(cfg, variant: str, train: list, val: list, bank: CodeBank, seed: int,
                init: TemporalGNN | None = None, epochs=None, lr=None) -> TemporalGNN:
    m = cfg["model"]
    set_seed(seed)
    dev = m["device"]
    flags = dataclasses.replace(VARIANTS[variant], **m.get("flag_overrides", {}).get(variant, {}))
    model = copy.deepcopy(init) if init is not None else TemporalGNN(m, flags)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or m["lr"], weight_decay=m["weight_decay"])
    val0 = [s for s in val if s["delta"] == 0] or val
    probs = contract_balanced_probs(train)
    n_epoch = min(len(train), m.get("epoch_samples", len(train)))
    best, best_ap, bad = None, -np.inf, 0
    for ep in range(epochs or m["epochs"]):
        model.train()
        # every contract weighs the same per epoch: a few giant sweeper campaigns must not
        # teach the model "this exact contract = scam" instead of transferable behaviour
        order = np.random.choice(len(train), n_epoch, replace=True, p=probs)
        for i in range(0, len(order), m["batch_size"]):
            batch = [train[j] for j in order[i:i + m["batch_size"]]]
            for attempt in range(3):
                try:
                    b = collate(batch, bank, dev)
                    loss = _loss(model(b), b["y"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite loss")
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    break
                except (RuntimeError, getattr(torch, "AcceleratorError", RuntimeError)) as e:
                    # macOS may abort long GPU command buffers ("Impacting Interactivity");
                    # the step's outputs are garbage then, so redo it rather than crash
                    log.warning("step failed (%s), retry %d", str(e)[:80], attempt)
                    opt.zero_grad()
        yv = np.array([s["y"] for s in val0])
        pv = np.clip(predict(model, val0, bank, device=dev)["p"], 1e-6, 1 - 1e-6)
        ap = average_precision_score(yv, pv)
        # AP saturates at 1.0 quickly on this task, so model selection uses class-balanced
        # validation log-loss, which keeps distinguishing well-calibrated from lucky epochs
        w = np.where(yv == 1, 0.5 / max(yv.mean(), 1e-6), 0.5 / max(1 - yv.mean(), 1e-6))
        vloss = float(-(w * (yv * np.log(pv) + (1 - yv) * np.log(1 - pv))).mean())
        log.info("%s seed %d epoch %d val AP %.4f val loss %.4f", variant, seed, ep, ap, vloss)
        score = -vloss
        if score > best_ap + 1e-4:
            best, best_ap, bad = copy.deepcopy(model.state_dict()), score, 0
        else:
            bad += 1
            if bad >= m["patience"]:
                break
    model.load_state_dict(best)
    return model


def contract_balanced_probs(samples) -> np.ndarray:
    """Sampling probabilities giving every (contract, class) group equal total weight."""
    keys = pd.Series([(int(s["nodes"][s["centres"][2]]), s["y"]) for s in samples])
    w = 1.0 / keys.map(keys.value_counts()).values
    return w / w.sum()
