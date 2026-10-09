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
from .model import VARIANTS, CodeEncoder, DelegNet
from .patterns import code_idioms
from .graph import NODE_FEATS
from .patterns.motifs import MOTIF_NAMES, motif_vector

# features that exist only because earlier delegators of the same contract were observed
SIBLING_MOTIFS = [MOTIF_NAMES.index(n) for n in MOTIF_NAMES if n.startswith("sib_")] + \
    [MOTIF_NAMES.index("sink_fanout"), MOTIF_NAMES.index("sponsor_funded_by_sink")]
SIBLING_NODE_FEATS = [NODE_FEATS.index(n) for n in ("delegated_to", "new_sponsor", "in_cnt", "new_cp")]

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
        self.variants = {}      # (chain, node) -> mutated code variants used as training augmentation
        self.augment = False
        self.evidence_dropout = 0.0   # share of training rows whose sibling evidence is hidden
        self.encoder = None     # frozen pre-trained code encoder (kept on the CPU)
        self.emb = {}           # token-sequence bytes -> embedding
        self.memory = []        # code-pattern memory: (family, 4-gram set, label) of training contracts
        self.sims = {}
        self._families = {}

    # ---------------------------------------------------------------- code-pattern retrieval memory
    def family(self, chain, node):
        if chain not in self._families:
            from .graph import code_families
            g = self.graphs[chain]
            nodes = np.array(sorted(g.code_tokens))
            self._families[chain] = dict(zip(nodes.tolist(), code_families(g, nodes).tolist())) if len(nodes) else {}
        return self._families[chain].get(node, -1)

    @staticmethod
    def _grams(tok):
        tok = tok.tolist()
        return frozenset(zip(tok, tok[1:], tok[2:], tok[3:]))

    def build_memory(self, samples):
        """Remember the canonical code patterns of the training contracts with their labels."""
        per = {}
        for s in samples:
            key = (s["chain"], int(s["nodes"][s["centres"][2]]))
            per.setdefault(key, []).append(s["y"])
        self.memory, self.sims = [], {}
        for (chain, node), ys in per.items():
            g = self.graphs[chain]
            if node in g.code_tokens:
                self.memory.append((self.family(chain, node), self._grams(g.code_tokens[node]), float(np.mean(ys))))

    def pattern_sims(self, chain, node) -> tuple[float, float]:
        """Best similarity to a remembered scam / benign pattern, skipping the contract's own code
        family, so training rows see what an unseen (zero-day) contract sees at test time."""
        if not self.memory:
            return 0.0, 0.0
        got = self.get(chain, node)
        if got is None:
            return 0.0, 0.0
        key = (chain, node, got[0].tobytes())
        if key not in self.sims:
            fam, g = self.family(chain, node), self._grams(got[0])
            best = [0.0, 0.0]
            for f, ref, y in self.memory:
                if f == fam:
                    continue
                j = len(g & ref) / max(len(g | ref), 1)
                c = 0 if y >= 0.5 else 1
                best[c] = max(best[c], j)
            self.sims[key] = tuple(best)
        return self.sims[key]

    def set_encoder(self, encoder):
        """Use a frozen code encoder whose embeddings are computed once per distinct code on the
        CPU. Long transformer kernels on the Apple GPU can hang under the macOS GPU watchdog, and
        fine-tuning a transformer on ~40 training contracts per fold would overfit anyway."""
        import copy as _copy
        self.encoder = _copy.deepcopy(encoder).cpu().eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.emb = {}

    @torch.no_grad()
    def embeddings(self, token_arrays):
        missing = [t for t in {t.tobytes(): t for t in token_arrays}.values() if t.tobytes() not in self.emb]
        for i in range(0, len(missing), 16):
            chunk = missing[i:i + 16]
            L = max(len(t) for t in chunk)
            tok = np.zeros((len(chunk), L), dtype=np.int64)
            for j, t in enumerate(chunk):
                tok[j, :len(t)] = t
            out = self.encoder(torch.as_tensor(tok))
            for t, e in zip(chunk, out):
                self.emb[t.tobytes()] = e
        return torch.stack([self.emb[t.tobytes()] for t in token_arrays]) if token_arrays else None

    def get(self, chain, node):
        if (chain, node) in self.override:
            return self.override[(chain, node)]
        if self.augment and (chain, node) in self.variants and random.random() < 0.5:
            return random.choice(self.variants[(chain, node)])
        g = self.graphs[chain]
        if node not in g.code_tokens:
            return None
        return g.code_tokens[node], g.code_idioms[node]

    def prepare_augmentation(self, samples, k=3, cache: Path | None = None):
        """Disguised copies of every training contract, so the model cannot memorise exact bytecode.

        Mutating large contracts is slow, so variants are built once per contract and cached on
        disk; every fold, seed and model reuses them (training rows only ever look up their own
        contracts, so sharing the cache across folds leaks nothing)."""
        if cache is not None and cache.exists() and not self.variants:
            self.variants = pickle.loads(cache.read_bytes())
        keys = {(s["chain"], int(s["nodes"][s["centres"][2]])) for s in samples}
        todo = [k_ for k_ in sorted(keys) if k_ not in self.variants and k_[1] in self.graphs[k_[0]].code_bytes]
        rng = random.Random(0)
        for n, key in enumerate(todo):
            raw0 = self.graphs[key[0]].code_bytes[key[1]]
            out = []
            for _ in range(k):
                raw = mutate_bytecode(raw0, rng, rng.uniform(0.5, 3.0))
                out.append((encode_tokens(code_idioms.tokens(raw)), code_idioms.idioms(raw)))
            self.variants[key] = out
            if n % 25 == 0:
                log.info("code augmentation: %d/%d new contracts", n + 1, len(todo))
        if todo and cache is not None:
            cache.write_bytes(pickle.dumps(self.variants, protocol=pickle.HIGHEST_PROTOCOL))

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
    extra = {}
    if bank.encoder is not None and codes:
        extra["code_emb"] = bank.embeddings([c[0] for c in codes]).to(device)
    motifs = np.stack([s["motifs"] for s in samples])
    xs_cat = np.concatenate(xs)
    if bank.augment and bank.evidence_dropout > 0:
        # evidence dropout: hide everything derived from earlier delegators of the same contract
        # for a random subset, so the model also learns to judge first victims (no sibling history)
        hide = np.random.rand(len(samples)) < bank.evidence_dropout
        motifs[np.ix_(hide, SIBLING_MOTIFS)] = 0.0
        offs = np.cumsum([0] + [len(s["nodes"]) for s in samples[:-1]])
        rows = (offs + np.array([s["centres"][2] for s in samples]))[hide]
        xs_cat[np.ix_(rows, SIBLING_NODE_FEATS)] = 0.0
    extra["pattern_sims"] = torch.as_tensor(
        [bank.pattern_sims(s["chain"], int(s["nodes"][s["centres"][2]])) for s in samples],
        dtype=torch.float32, device=device)
    return dict(**extra,
        x=T(xs_cat), role=T(roles, torch.long), is_contract=T(kinds, torch.long), code_slot=T(slots, torch.long),
        code_tokens=torch.as_tensor(tok, device=device),
        code_idioms=torch.as_tensor(np.stack([c[1] for c in codes]) if codes else np.zeros((0, 21), np.float32),
                                    device=device),
        esrc=T(es, torch.long), edst=T(ed, torch.long), etype=T(et, torch.long), edt=T(edt), efeat=T(ef),
        centres=torch.as_tensor(np.stack(cen), device=device),
        motifs=torch.as_tensor(motifs, device=device),
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
    if hasattr(model, "code_enc") and bank.encoder is None:   # e.g. a reloaded model in the analyses
        bank.set_encoder(model.code_enc)
    p, z, d2 = [], [], []
    for i in range(0, len(samples), bs):
        out = _forward_safe(model, samples[i:i + bs], bank, device)
        p.append(torch.sigmoid(out["logit"])); z.append(out["z"])
        if "d2" in out:
            d2.append(out["d2"])
    res = dict(p=torch.cat(p).numpy(), z=torch.cat(z).numpy())
    if d2:
        res["d2"] = torch.cat(d2).numpy()
    elif hasattr(model, "post_protos"):
        P = model.post_protos.cpu().numpy()
        res["d2"] = ((res["z"][:, None, :] - P[None]) ** 2).sum(-1)
    return res


def _loss(model, out, y, m):
    pos = (y == 0).sum().clamp_min(1) / (y == 1).sum().clamp_min(1)
    loss = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos)
    if "d2" in out:
        # prototype module: its own classification loss (so prototypes stay class-meaningful
        # for explanations) plus ProtoPNet clustering / separation / diversity terms
        ploss = F.binary_cross_entropy_with_logits(out["proto_logit"], y, pos_weight=pos)
        cl, sep, div = model.proto.losses(out["d2"], y)
        loss = loss + m.get("lambda_proto", 0.3) * ploss + m["lambda_cluster"] * cl \
            + m["lambda_sep"] * sep + m["lambda_div"] * div
    return loss


def train_model(cfg, variant: str, train: list, val: list, bank: CodeBank, seed: int,
                init: DelegNet | None = None, epochs=None, lr=None) -> DelegNet:
    m = cfg["model"]
    set_seed(seed)
    dev = m["device"]
    flags = dataclasses.replace(VARIANTS[variant], **m.get("flag_overrides", {}).get(variant, {}))
    model = copy.deepcopy(init) if init is not None else DelegNet(m, flags)
    enc_path = Path(cfg["run_root"]) / "code_encoder.pt"
    if init is None and hasattr(model, "code_enc") and enc_path.exists():
        state = torch.load(enc_path, map_location="cpu")
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError(f"{enc_path} contains non-finite weights; rerun `run.py pretrain`")
        model.code_enc.load_state_dict(state)
    if hasattr(model, "code_enc"):
        for p in model.code_enc.parameters():
            p.requires_grad_(False)
        bank.set_encoder(model.code_enc)
    if model.flags.memory and init is None:
        bank.build_memory(train)
    bank.evidence_dropout = m.get("evidence_dropout", 0.0) if model.flags.evidence_dropout else 0.0
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or m["lr"], weight_decay=m["weight_decay"])
    val0 = [s for s in val if s["delta"] == 0] or val
    probs = contract_balanced_probs(train)
    n_epoch = min(len(train), m.get("epoch_samples", len(train)))
    if model.flags.code != "none":
        bank.prepare_augmentation(train, cache=Path(cfg["data_root"]) / "code_variants.pkl")
    best, best_ap, bad = None, -np.inf, 0
    for ep in range(epochs or m["epochs"]):
        model.train()
        bank.augment = model.flags.code != "none"
        # every contract weighs the same per epoch: a few giant sweeper campaigns must not
        # teach the model "this exact contract = scam" instead of transferable behaviour
        order = np.random.choice(len(train), n_epoch, replace=True, p=probs)
        for i in range(0, len(order), m["batch_size"]):
            batch = [train[j] for j in order[i:i + m["batch_size"]]]
            for attempt in range(3):
                try:
                    b = collate(batch, bank, dev)
                    loss = _loss(model, model(b), b["y"], m)
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
        bank.augment = False
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
    if model.flags.use_proto:
        _project_and_tune(model, train, bank, m, dev)
    else:
        _fit_posthoc_prototypes(model, train, bank, m, dev)
    return model


def _kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    """k-means++ / Lloyd in NumPy (scikit-learn's KMeans segfaults next to torch's OpenMP on macOS)."""
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None] - np.array(centres)[None]) ** 2).sum(-1), 1)
        centres.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(centres)
    for _ in range(iters):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c


def _fit_posthoc_prototypes(model, train, bank, m, dev):
    """Prototypes for explanation / novelty when they are not trained jointly: per-class k-means
    centres of the training embeddings, each snapped to its nearest real training delegation."""
    base = [s for s in train if s["delta"] == 0]
    z = predict(model, base, bank, device=dev)["z"]
    y = np.array([s["y"] for s in base])
    ids = np.array([s["auth_id"] for s in base])
    protos, cls, src = [], [], []
    for c in (0, 1):
        zc, idc = z[y == c], ids[y == c]
        if not len(zc):
            continue
        k = min(m["prototypes_per_class"], len(zc))
        for centre in _kmeans(zc, k):
            j = int(((zc - centre) ** 2).sum(1).argmin())
            protos.append(zc[j]); cls.append(c); src.append(int(idc[j]))
    model.register_buffer("post_protos", torch.as_tensor(np.stack(protos), dtype=torch.float32))
    model.register_buffer("post_proto_class", torch.as_tensor(cls))
    model.register_buffer("post_proto_source", torch.as_tensor(src))


def contract_balanced_probs(samples) -> np.ndarray:
    """Sampling probabilities giving every (contract, class) group equal total weight."""
    keys = pd.Series([(int(s["nodes"][s["centres"][2]]), s["y"]) for s in samples])
    w = 1.0 / keys.map(keys.value_counts()).values
    return w / w.sum()


def _project_and_tune(model, train, bank, m, dev):
    """ProtoPNet step: snap prototypes to real training delegations, then refit only the
    prototype->class weights so every prediction is explained by real examples."""
    base = [s for s in train if s["delta"] == 0]
    out = predict(model, base, bank, device=dev)
    model.proto.project(torch.as_tensor(out["z"], device=dev), torch.tensor([s["y"] for s in base], device=dev),
                        torch.tensor([s["auth_id"] for s in base], device=dev))
    # prototypes now explain the prediction head's embedding; no prediction weights change here


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
    dev = m["device"]
    cd = m.get("code_dim", m["dim"])
    enc = CodeEncoder(cd, m["code_layers"], m["heads"], m["code_len"], m["dropout"]).to(dev)
    head = torch.nn.Sequential(torch.nn.Linear(cd, cd), torch.nn.GELU(), torch.nn.Linear(cd, 64)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=5e-4)
    rng, prng = np.random.default_rng(0), random.Random(0)
    # disguised copies are expensive to make, so build a pool once and sample from it
    pool = {}
    for i, raw in enumerate(raws):
        pool[i] = [encode_tokens(code_idioms.tokens(raw), m["code_len"])] + \
                  [encode_tokens(code_idioms.tokens(mutate_bytecode(raw, prng, prng.uniform(0.5, 3))), m["code_len"])
                   for _ in range(4)]
    idx = list(range(len(raws)))

    def view(i):
        return _augment(pool[i][prng.randrange(len(pool[i]))], rng)

    def pad(seqs):
        out = np.zeros((len(seqs), max(map(len, seqs))), dtype=np.int64)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
        return torch.as_tensor(out, device=dev)

    bs = 128
    for ep in range(m["pretrain_epochs"]):
        prng.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx) - 1, bs):
            batch = idx[i:i + bs]
            if len(batch) < 2:
                continue
            z1 = F.normalize(head(enc(pad([view(r) for r in batch]))), dim=1)
            z2 = F.normalize(head(enc(pad([view(r) for r in batch]))), dim=1)
            z = torch.cat([z1, z2])
            sim = z @ z.t() / 0.1
            sim.fill_diagonal_(-1e9)
            n = len(batch)
            target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(dev)
            loss = F.cross_entropy(sim, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"code pre-training diverged (loss {loss.item()}) at epoch {ep}")
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        log.info("pretrain epoch %d loss %.4f", ep, tot)
    Path(cfg["run_root"]).mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in enc.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError("code encoder has non-finite weights; refusing to save")
    torch.save(state, Path(cfg["run_root"]) / "code_encoder.pt")
