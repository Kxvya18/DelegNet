from __future__ import annotations

import copy
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
from .model import VARIANTS, CodeEncoder, DelegNet
from .patterns import code_idioms
from .patterns.motifs import motif_vector

log = logging.getLogger("train")
_G = None  # graph shared with forked workers


# ------------------------------------------------------------------ samples

def _make(args):
    auth_id, event, delta, label, gcfg, mask = args
    g = _G
    cutoff = g.cutoff_for(event, delta)
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


def collate(samples: list[dict], bank: CodeBank, device="cpu") -> dict:
    xs, roles, kinds, slots = [], [], [], []
    es, ed, et, edt, ef, cen = [], [], [], [], [], []
    code_keys, code_idx = [], {}
    off = 0
    for s in samples:
        xs.append(s["x"]); roles.append(s["role"]); kinds.append(s["is_contract"])
        slot = np.full(len(s["nodes"]), -1, dtype=np.int64)
        for i in np.flatnonzero(s["is_contract"]):
            k = (s["chain"], int(s["nodes"][i]))
            if k not in code_idx and bank.get(*k) is not None:
                code_idx[k] = len(code_keys)
                code_keys.append(k)
            slot[i] = code_idx.get(k, -1)
        slots.append(slot)
        es.append(s["esrc"] + off); ed.append(s["edst"] + off); et.append(s["etype"])
        edt.append(s["edt"]); ef.append(s["efeat"]); cen.append(s["centres"] + off)
        off += len(s["nodes"])
    codes = [bank.get(*k) for k in code_keys]
    L = max([len(c[0]) for c in codes], default=1)
    tok = np.zeros((len(codes), L), dtype=np.int64)
    for i, c in enumerate(codes):
        tok[i, :len(c[0])] = c[0]
    T = lambda a, dt=torch.float32: torch.as_tensor(np.concatenate(a) if isinstance(a, list) else a,  # noqa: E731
                                                     dtype=dt, device=device)
    return dict(
        x=T(xs), role=T(roles, torch.long), is_contract=T(kinds, torch.long), code_slot=T(slots, torch.long),
        code_tokens=torch.as_tensor(tok, device=device),
        code_idioms=torch.as_tensor(np.stack([c[1] for c in codes]) if codes else np.zeros((0, 21), np.float32),
                                    device=device),
        esrc=T(es, torch.long), edst=T(ed, torch.long), etype=T(et, torch.long), edt=T(edt), efeat=T(ef),
        centres=torch.as_tensor(np.stack(cen), device=device),
        motifs=torch.as_tensor(np.stack([s["motifs"] for s in samples]), device=device),
        authf=torch.as_tensor(np.stack([s["authf"] for s in samples]), device=device),
        y=torch.as_tensor([s["y"] for s in samples], dtype=torch.float32, device=device),
    )


# ------------------------------------------------------------------ training

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def predict(model, samples, bank, bs=128, device="cpu") -> dict:
    model.eval()
    p, z, d2 = [], [], []
    for i in range(0, len(samples), bs):
        out = model(collate(samples[i:i + bs], bank, device))
        p.append(torch.sigmoid(out["logit"]).cpu()); z.append(out["z"].cpu())
        if "d2" in out:
            d2.append(out["d2"].cpu())
    res = dict(p=torch.cat(p).numpy(), z=torch.cat(z).numpy())
    if d2:
        res["d2"] = torch.cat(d2).numpy()
    return res


def _loss(model, out, y, m):
    pos = (y == 0).sum().clamp_min(1) / (y == 1).sum().clamp_min(1)
    loss = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos)
    if "d2" in out:
        cl, sep, div = model.proto.losses(out["d2"], y)
        loss = loss + m["lambda_cluster"] * cl + m["lambda_sep"] * sep + m["lambda_div"] * div
    return loss


def train_model(cfg, variant: str, train: list, val: list, bank: CodeBank, seed: int,
                init: DelegNet | None = None, epochs=None, lr=None) -> DelegNet:
    m = cfg["model"]
    set_seed(seed)
    dev = m["device"]
    model = copy.deepcopy(init) if init is not None else DelegNet(m, VARIANTS[variant])
    enc_path = Path(cfg["run_root"]) / "code_encoder.pt"
    if init is None and hasattr(model, "code_enc") and enc_path.exists() and "noPretrain" not in variant:
        model.code_enc.load_state_dict(torch.load(enc_path, map_location="cpu"))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or m["lr"], weight_decay=m["weight_decay"])
    val0 = [s for s in val if s["delta"] == 0] or val
    best, best_ap, bad = None, -1, 0
    for ep in range(epochs or m["epochs"]):
        model.train()
        order = np.random.permutation(len(train))
        for i in range(0, len(order), m["batch_size"]):
            b = collate([train[j] for j in order[i:i + m["batch_size"]]], bank, dev)
            loss = _loss(model, model(b), b["y"], m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ap = average_precision_score([s["y"] for s in val0], predict(model, val0, bank, device=dev)["p"])
        log.info("%s seed %d epoch %d val AP %.4f", variant, seed, ep, ap)
        if ap > best_ap + 1e-4:
            best, best_ap, bad = copy.deepcopy(model.state_dict()), ap, 0
        else:
            bad += 1
            if bad >= m["patience"]:
                break
    model.load_state_dict(best)
    if model.flags.use_proto:
        _project_and_tune(model, train, bank, m, dev)
    return model


def _project_and_tune(model, train, bank, m, dev):
    """ProtoPNet step: snap prototypes to real training delegations, then refit only the
    prototype->class weights so every prediction is explained by real examples."""
    base = [s for s in train if s["delta"] == 0]
    out = predict(model, base, bank, device=dev)
    model.proto.project(torch.as_tensor(out["z"], device=dev), torch.tensor([s["y"] for s in base], device=dev),
                        torch.tensor([s["auth_id"] for s in base], device=dev))
    for p in model.parameters():
        p.requires_grad_(False)
    model.proto.last.requires_grad_(True)
    opt = torch.optim.Adam([model.proto.last], lr=m["lr"])
    for _ in range(5):
        order = np.random.permutation(len(train))
        for i in range(0, len(order), m["batch_size"]):
            b = collate([train[j] for j in order[i:i + m["batch_size"]]], bank, dev)
            o = model(b)
            loss = F.binary_cross_entropy_with_logits(o["logit"], b["y"]) + 1e-4 * model.proto.last.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)


# ------------------------------------------------------------------ code pre-training

def _augment(tokens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = tokens[1:].copy()
    if len(t) > 32 and rng.random() < 0.5:           # random crop
        L = int(len(t) * rng.uniform(0.6, 1.0))
        s = rng.integers(0, len(t) - L + 1)
        t = t[s:s + L]
    t[rng.random(len(t)) < 0.15] = 2                  # mask to <unk>
    return np.concatenate([[1], t])


def pretrain_code_encoder(cfg):
    """SimCLR on delegate bytecode: two views = independent bytecode mutations followed by
    token-level crop/mask. Uses only contracts first seen in the source chain's training period."""
    m = cfg["model"]
    src = cfg["experiment"]["source_chain"]
    g, tg = load_graph(cfg, src)
    cut = np.quantile(g.ts[tg.event.values], cfg["experiment"]["split"][0])
    nodes = tg[g.ts[tg.event.values] <= cut].contract_node.unique()
    raws = [g.code_bytes[n] for n in nodes if n in g.code_bytes]
    log.info("pre-training code encoder on %d contracts", len(raws))
    set_seed(0)
    enc = CodeEncoder(m["dim"], m["code_layers"], m["heads"], m["code_len"], m["dropout"])
    head = torch.nn.Sequential(torch.nn.Linear(m["dim"], m["dim"]), torch.nn.GELU(), torch.nn.Linear(m["dim"], 64))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=5e-4)
    rng, prng = np.random.default_rng(0), random.Random(0)

    def view(raw):
        mutated = mutate_bytecode(raw, prng, prng.uniform(0, 3))
        return _augment(encode_tokens(code_idioms.tokens(mutated), m["code_len"]), rng)

    def pad(seqs):
        out = np.zeros((len(seqs), max(map(len, seqs))), dtype=np.int64)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
        return torch.as_tensor(out)

    bs = 128
    for ep in range(m["pretrain_epochs"]):
        prng.shuffle(raws)
        tot = 0.0
        for i in range(0, len(raws) - 1, bs):
            batch = raws[i:i + bs]
            if len(batch) < 2:
                continue
            z1 = F.normalize(head(enc(pad([view(r) for r in batch]))), dim=1)
            z2 = F.normalize(head(enc(pad([view(r) for r in batch]))), dim=1)
            z = torch.cat([z1, z2])
            sim = z @ z.t() / 0.1
            sim.fill_diagonal_(-1e9)
            n = len(batch)
            target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)])
            loss = F.cross_entropy(sim, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        log.info("pretrain epoch %d loss %.4f", ep, tot)
    Path(cfg["run_root"]).mkdir(exist_ok=True)
    torch.save(enc.state_dict(), Path(cfg["run_root"]) / "code_encoder.pt")
