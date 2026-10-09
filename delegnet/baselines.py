"""Non-neural baselines.

  HashBlocklist  what wallets and explorers do today: exact bytecode match against
                 contracts already known to be malicious at training time
  OpcodeKNN      fuzzy-hash analogue: Jaccard similarity of canonical opcode 4-grams to
                 the nearest known malicious vs benign contract
  XGBoost        strong tabular model over the same centre-node, motif, auth and idiom
                 features DelegNet sees (everything except message passing)
"""
from __future__ import annotations

import numpy as np

from . import xgb_worker


def _code_key(bank, s):
    c = s["nodes"][s["centres"][2]]
    got = bank.get(s["chain"], int(c))
    return (s["chain"], int(c)), got


class HashBlocklist:
    name = "HashBlocklist"

    def fit(self, train, val, bank):
        self.known = {}
        for s in train:
            _, got = _code_key(bank, s)
            if got is not None:
                self.known.setdefault(got[0].tobytes(), []).append(s["y"])
        self.known = {k: np.mean(v) for k, v in self.known.items()}
        return self

    def predict(self, samples, bank):
        out = []
        for s in samples:
            _, got = _code_key(bank, s)
            out.append(self.known.get(got[0].tobytes(), 0.5) if got is not None else 0.5)
        return np.array(out)


class OpcodeKNN:
    name = "OpcodeKNN"

    @staticmethod
    def grams(tok):
        return set(zip(tok, tok[1:], tok[2:], tok[3:]))

    def fit(self, train, val, bank):
        ref = {}
        for s in train:
            k, got = _code_key(bank, s)
            if got is not None:
                ref.setdefault(k, [self.grams(got[0].tolist()), []])[1].append(s["y"])
        self.ref = [(g, np.mean(ys)) for g, ys in ref.values()]
        return self

    def similarities(self, samples, bank) -> np.ndarray:
        """[best similarity to a known-malicious contract, to a known-benign one] per sample."""
        cache, out = {}, []
        for s in samples:
            k, got = _code_key(bank, s)
            if got is None:
                out.append((0.0, 0.0))
                continue
            key = (k, got[0].tobytes())
            if key not in cache:
                g = self.grams(got[0].tolist())
                best_m = max((len(g & r) / max(len(g | r), 1) for r, y in self.ref if y >= 0.5), default=0)
                best_b = max((len(g & r) / max(len(g | r), 1) for r, y in self.ref if y < 0.5), default=0)
                cache[key] = (best_m, best_b)
            out.append(cache[key])
        return np.array(out, dtype=np.float32)

    def predict(self, samples, bank):
        sim = self.similarities(samples, bank)
        return 0.5 + 0.5 * (sim[:, 0] - sim[:, 1])


class OpcodeKNNRaw(OpcodeKNN):
    """Prior-art fuzzy hashing: opcode 4-gram similarity on the *raw* deployed bytecode, without
    DelegNet's canonicalisation (so junk instructions and equivalent-opcode swaps count)."""
    name = "OpcodeKNN-raw"

    @staticmethod
    def _ops(bank, s):
        from .evm import opcode_sequence
        raw = bank.raw(s["chain"], int(s["nodes"][s["centres"][2]]))
        return None if raw is None else opcode_sequence(raw)

    def fit(self, train, val, bank):
        ref = {}
        for s in train:
            k = (s["chain"], int(s["nodes"][s["centres"][2]]))
            ops = self._ops(bank, s)
            if ops is not None:
                ref.setdefault(k, [self.grams(ops), []])[1].append(s["y"])
        self.ref = [(g, np.mean(ys)) for g, ys in ref.values()]
        return self

    def similarities(self, samples, bank) -> np.ndarray:
        cache, out = {}, []
        for s in samples:
            ops = self._ops(bank, s)
            if ops is None:
                out.append((0.0, 0.0))
                continue
            key = hash(tuple(ops))
            if key not in cache:
                g = self.grams(ops)
                best_m = max((len(g & r) / max(len(g | r), 1) for r, y in self.ref if y >= 0.5), default=0)
                best_b = max((len(g & r) / max(len(g | r), 1) for r, y in self.ref if y < 0.5), default=0)
                cache[key] = (best_m, best_b)
            out.append(cache[key])
        return np.array(out, dtype=np.float32)


def tabular(samples, bank) -> np.ndarray:
    rows = []
    for s in samples:
        c = s["centres"]
        _, got = _code_key(bank, s)
        idi = got[1] if got is not None else np.zeros(21, np.float32)
        rows.append(np.concatenate([s["x"][c[0]], s["x"][c[1]], s["x"][c[2]], s["motifs"], s["authf"], idi]))
    return np.stack(rows)


class XGBoost:
    name = "XGBoost"

    def __init__(self, seed=0, params=None):
        self.seed, self.params = seed, params or {}

    def fit(self, train, val, bank):
        X, y = tabular(train, bank), np.array([s["y"] for s in train])
        v0 = [s for s in val if s["delta"] == 0] or val
        Xv, yv = tabular(v0, bank), np.array([s["y"] for s in v0])
        self.raw = xgb_worker.call(xgb_worker.fit, X, y, Xv, yv, self.seed, self.params)
        return self

    def predict(self, samples, bank):
        return xgb_worker.call(xgb_worker.predict, self.raw, tabular(samples, bank))

    def leaves(self, samples, bank):
        return xgb_worker.call(xgb_worker.predict, self.raw, tabular(samples, bank), True)


BASELINES = {"HashBlocklist": HashBlocklist, "OpcodeKNN": OpcodeKNN, "OpcodeKNN-raw": OpcodeKNNRaw,
             "XGBoost": XGBoost}
