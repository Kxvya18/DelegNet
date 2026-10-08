"""Real on-chain collection of EIP-7702 delegations and the activity around them.

Stages (each resumable; output under data/<chain>/):
  scan      eth_getBlockByNumber over a range, keep type-4 txs, recover each authority
  code      eth_getCode for delegate contracts and for counterparties we need to type
  activity  native value transfers (including internal calls) via trace_filter and
            ERC-20 Transfer logs via eth_getLogs, for a set of tracked addresses
  senders   tx.from for every transfer tx, so we can tell self-initiated outflows from
            outflows triggered by someone else through the delegated code

Different providers expose different methods, so endpoints are routed per method.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path

import aiohttp
import pandas as pd
import rlp
from eth_hash.auto import keccak
from eth_keys import keys

log = logging.getLogger("collect")
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ZERO = "0x" + "0" * 40
DELEGATION_PREFIX = "0xef0100"
RANGE_ERRORS = ("range", "too many", "limit", "exceed", "too large", "timeout", "response size")
NOT_ON_PLAN = ("plan", "upgrade", "archive", "personal token")


DROPPED: list = []   # (lo, hi, error) ranges that could not be fetched


class RPCError(Exception):
    pass


class RPC:
    def __init__(self, endpoints: dict, concurrency: int = 8, timeout: int = 90, retries: int = 6):
        self.endpoints = {k: v if isinstance(v, list) else [v] for k, v in endpoints.items()}
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retries = retries
        self.session: aiohttp.ClientSession | None = None
        self._rr = 0

    async def __aenter__(self):
        # some providers answer with brotli, which aiohttp can't decode without extras
        self.session = aiohttp.ClientSession(timeout=self.timeout, headers={"Accept-Encoding": "gzip"})
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    def _pick(self, urls, attempt):
        self._rr += 1  # spread load over every endpoint, not just on retries
        return urls[(self._rr + attempt) % len(urls)]

    async def batch(self, method: str, params_list: list[list]) -> list:
        """JSON-RPC batch over endpoints that support it; items that fail are retried singly."""
        urls = self.endpoints.get(method + ":batch") or self.endpoints.get(method, self.endpoints["default"])
        payload = [{"jsonrpc": "2.0", "id": i, "method": method, "params": p} for i, p in enumerate(params_list)]
        out = [None] * len(params_list)
        delay = 1.0
        for attempt in range(self.retries):
            url = self._pick(urls, attempt)
            try:
                async with self.sem, self.session.post(url, json=payload) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    body = json.loads(await resp.text())
                if isinstance(body, list):
                    for item in body:
                        if isinstance(item, dict) and item.get("result") is not None:
                            out[item["id"]] = item["result"]
                    break
                raise aiohttp.ClientError("batch not supported")
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, KeyError, TypeError):
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 30)
        for i, r in enumerate(out):
            if r is None:
                out[i] = await self.call(method, params_list[i])
        return out

    async def call(self, method: str, params: list):
        urls = self.endpoints.get(method, self.endpoints["default"])
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        delay = 1.0
        for attempt in range(self.retries):
            url = self._pick(urls, attempt)
            try:
                async with self.sem, self.session.post(url, json=payload) as resp:
                    text = await resp.text()
                    if resp.status == 429 or resp.status >= 500:
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    body = json.loads(text)
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                log.debug("%s retry %d: %s", method, attempt, e)
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 30)
                continue
            if "error" in body:
                err = body["error"] if isinstance(body["error"], dict) else {"message": str(body["error"])}
                body["error"] = err
                msg = str(err.get("message", "")).lower()
                if "rate" in msg or err.get("code") in (-32005, 429):
                    await asyncio.sleep(delay + random.random())
                    delay = min(delay * 2, 30)
                    continue
                raise RPCError(f"{method}: {body['error']}")
            return body["result"]
        raise RPCError(f"{method}: gave up after {self.retries} attempts")


# ---------------------------------------------------------------- authorizations

def recover_authority(chain_id: int, address: str, nonce: int, y: int, r: int, s: int) -> str | None:
    """EIP-7702: authority = ecrecover(keccak(0x05 || rlp([chain_id, address, nonce])), y, r, s)."""
    try:
        msg = keccak(b"\x05" + rlp.encode([chain_id, bytes.fromhex(address[2:]), nonce]))
        sig = keys.Signature(vrs=(y, r, s))
        return sig.recover_public_key_from_msg_hash(msg).to_checksum_address().lower()
    except Exception:  # malformed or high-s signatures are rejected by the protocol too
        return None


def parse_block(block: dict, chain_id: int) -> list[dict]:
    rows = []
    bn, ts = int(block["number"], 16), int(block["timestamp"], 16)
    for tx in block["transactions"]:
        if tx.get("type") != "0x4":
            continue
        auths = tx.get("authorizationList") or []
        for k, a in enumerate(auths):
            a_chain, nonce = int(a["chainId"], 16), int(a["nonce"], 16)
            delegate = a["address"].lower()
            authority = (a.get("authority") or "").lower() or recover_authority(
                a_chain, delegate, nonce, int(a.get("yParity", a.get("v", "0x0")), 16),
                int(a["r"], 16), int(a["s"], 16))
            rows.append(dict(
                block=bn, ts=ts, tx_index=int(tx["transactionIndex"], 16), tx_hash=tx["hash"],
                sponsor=tx["from"].lower(), tx_to=(tx.get("to") or "").lower(), auth_index=k,
                batch_size=len(auths), auth_chain_id=a_chain, auth_nonce=nonce,
                delegate=delegate, authority=authority,
                chain_ok=a_chain in (0, chain_id), revoke=delegate == ZERO))
    return rows


async def scan(rpc: RPC, out: Path, chain_id: int, start: int, end: int, chunk: int = 2000):
    out = out / "auth"
    out.mkdir(parents=True, exist_ok=True)

    async def group(bs):
        return await rpc.batch("eth_getBlockByNumber", [[hex(b), True] for b in bs])

    for lo in range(start, end + 1, chunk):
        hi = min(lo + chunk - 1, end)
        f = out / f"{lo:010d}_{hi:010d}.parquet"
        if f.exists():
            continue
        groups = [list(range(b, min(b + 10, hi + 1))) for b in range(lo, hi + 1, 10)]
        blocks = [b for res in await asyncio.gather(*(group(gr) for gr in groups)) for b in res]
        rows = [r for b in blocks if b for r in parse_block(b, chain_id)]
        pd.DataFrame(rows, columns=AUTH_COLUMNS).to_parquet(f, index=False)
        log.info("scan %d-%d: %d authorizations", lo, hi, len(rows))


AUTH_COLUMNS = ["block", "ts", "tx_index", "tx_hash", "sponsor", "tx_to", "auth_index", "batch_size",
                "auth_chain_id", "auth_nonce", "delegate", "authority", "chain_ok", "revoke"]


# ---------------------------------------------------------------- code

async def fetch_code(rpc: RPC, out: Path, addresses: list[str], fname: str = "code.parquet"):
    f = out / fname
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["address", "code"])
    todo = sorted(set(addresses) - set(have.address))

    async def one(a):
        try:
            return a, await rpc.call("eth_getCode", [a, "latest"])
        except RPCError:
            return a, None

    for i in range(0, len(todo), 500):
        res = await asyncio.gather(*(one(a) for a in todo[i:i + 500]))
        have = pd.concat([have, pd.DataFrame(res, columns=["address", "code"]).dropna()])
        have.to_parquet(f, index=False)
        log.info("code: %d/%d", min(i + 500, len(todo)), len(todo))
    return have


# ---------------------------------------------------------------- activity

async def _split_range(fetch, lo: int, hi: int, min_span: int = 1):
    """Call fetch(lo, hi); on provider range/size errors, bisect the block range."""
    try:
        return await fetch(lo, hi)
    except RPCError as e:
        msg = str(e).lower()
        splittable = any(s in msg for s in RANGE_ERRORS)   # includes free-plan *timeouts*
        if hi - lo < min_span or not splittable or (any(s in msg for s in NOT_ON_PLAN) and "timeout" not in msg):
            # never silent: a dropped range is recorded so coverage can be reported and re-collected
            log.warning("dropping %d-%d: %s", lo, hi, e)
            DROPPED.append((lo, hi, str(e)[:200]))
            return []
        mid = (lo + hi) // 2
        a = await _split_range(fetch, lo, mid, min_span)
        b = await _split_range(fetch, mid + 1, hi, min_span)
        return a + b


def _pad(addr: str) -> str:
    return "0x" + "0" * 24 + addr[2:]


async def native_transfers(rpc: RPC, addrs: list[str], lo: int, hi: int) -> list[dict]:
    rows = []
    for side in ("fromAddress", "toAddress"):
        async def fetch(a, b, side=side):
            res, after = [], 0
            while True:
                page = await rpc.call("trace_filter", [{"fromBlock": hex(a), "toBlock": hex(b),
                                                        side: addrs, "after": after, "count": 1000}])
                res += page
                if len(page) < 1000:
                    return res
                after += 1000

        for t in await _split_range(fetch, lo, hi, min_span=50):
            if t.get("error"):
                continue
            act = t["action"]
            if t["type"] == "call" and act.get("callType") in ("call", None):
                src, dst, val = act["from"], act["to"], int(act.get("value", "0x0"), 16)
            elif t["type"] == "suicide":
                src, dst, val = act["address"], act["refundAddress"], int(act.get("balance", "0x0"), 16)
            else:
                continue
            if val == 0:
                continue
            rows.append(dict(block=t["blockNumber"], tx_hash=t["transactionHash"],
                             tx_index=t.get("transactionPosition", 0) or 0,
                             sub=len(t["traceAddress"]), src=src.lower(), dst=dst.lower(),
                             value=val / 1e18, asset="native", internal=len(t["traceAddress"]) > 0))
    return rows


async def token_transfers(rpc: RPC, addrs: list[str], lo: int, hi: int) -> list[dict]:
    rows, padded = [], [_pad(a) for a in addrs]
    for topics in ([TRANSFER_TOPIC, padded], [TRANSFER_TOPIC, None, padded]):
        async def fetch(a, b, topics=topics):
            return await rpc.call("eth_getLogs", [{"fromBlock": hex(a), "toBlock": hex(b), "topics": topics}])

        for lg in await _split_range(fetch, lo, hi, min_span=200):
            if len(lg["topics"]) != 3 or lg.get("removed"):
                continue  # ERC-721 has an indexed tokenId (4 topics)
            raw = int(lg["data"], 16) if lg["data"] not in ("0x", "") else 0
            if raw == 0:
                continue
            rows.append(dict(block=int(lg["blockNumber"], 16), tx_hash=lg["transactionHash"],
                             tx_index=int(lg["transactionIndex"], 16), sub=1000 + int(lg["logIndex"], 16),
                             src="0x" + lg["topics"][1][-40:], dst="0x" + lg["topics"][2][-40:],
                             value=float(raw), asset=lg["address"].lower(), internal=True))
    return rows


async def activity(rpc: RPC, out: Path, addrs: list[str], start: int, end: int, tag: str,
                   block_chunk: int = 20000, addr_chunk: int = 100, token_logs: bool = True):
    out = out / f"transfers_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    addrs = sorted(set(addrs))
    jobs = []
    for i in range(0, len(addrs), addr_chunk):
        for lo in range(start, end + 1, block_chunk):
            jobs.append((i, addrs[i:i + addr_chunk], lo, min(lo + block_chunk - 1, end)))

    async def run(job):
        i, group, lo, hi = job
        f = out / f"a{i:06d}_{lo:010d}.parquet"
        if f.exists():
            return
        before = len(DROPPED)
        rows = await native_transfers(rpc, group, lo, hi)
        if token_logs:
            rows += await token_transfers(rpc, group, lo, hi)
        if len(DROPPED) > before:   # incomplete: leave no file so a rerun retries this job
            log.warning("job a%06d_%d incomplete, will retry on rerun", i, lo)
            return
        pd.DataFrame(rows, columns=TRANSFER_COLUMNS).to_parquet(f, index=False)

    for k in range(0, len(jobs), 16):
        await asyncio.gather(*(run(j) for j in jobs[k:k + 16]))
        log.info("activity[%s]: %d/%d jobs", tag, min(k + 16, len(jobs)), len(jobs))


TRANSFER_COLUMNS = ["block", "tx_hash", "tx_index", "sub", "src", "dst", "value", "asset", "internal"]


async def senders(rpc: RPC, out: Path, tx_hashes: list[str]):
    f = out / "senders.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["tx_hash", "sender"])
    todo = sorted(set(tx_hashes) - set(have.tx_hash))

    async def group(hs):
        try:
            txs = await rpc.batch("eth_getTransactionByHash", [[h] for h in hs])
        except RPCError:
            return [(h, None) for h in hs]
        return [(h, tx["from"].lower() if tx else None) for h, tx in zip(hs, txs)]

    for i in range(0, len(todo), 1000):
        chunk = todo[i:i + 1000]
        res = [r for part in await asyncio.gather(*(group(chunk[j:j + 20]) for j in range(0, len(chunk), 20)))
               for r in part]
        have = pd.concat([have, pd.DataFrame(res, columns=["tx_hash", "sender"]).dropna()])
        have.to_parquet(f, index=False)
        log.info("senders: %d/%d", min(i + 1000, len(todo)), len(todo))


# ---------------------------------------------------------------- loading

def _read_dir(d: Path, columns) -> pd.DataFrame:
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        return pd.DataFrame(columns=columns)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_chain(out: Path) -> dict[str, pd.DataFrame]:
    auth = _read_dir(out / "auth", AUTH_COLUMNS)
    auth = auth[auth.authority.notna() & auth.chain_ok].astype(
        {"block": "int64", "ts": "int64", "tx_index": "int64", "auth_index": "int64", "batch_size": "int64"})
    auth = auth.reset_index(drop=True)
    tr = pd.concat([_read_dir(out / "transfers_hop1", TRANSFER_COLUMNS),
                    _read_dir(out / "transfers_hop2", TRANSFER_COLUMNS)], ignore_index=True)
    tr = tr.astype({"block": "int64", "tx_index": "int64", "sub": "int64", "value": "float64", "internal": "bool"})
    tr = tr.drop_duplicates(["tx_hash", "sub", "src", "dst", "asset", "value"]).reset_index(drop=True)
    snd = pd.read_parquet(out / "senders.parquet") if (out / "senders.parquet").exists() else \
        pd.DataFrame(columns=["tx_hash", "sender"])
    tr = tr.merge(snd, on="tx_hash", how="left")
    code = pd.read_parquet(out / "code.parquet") if (out / "code.parquet").exists() else \
        pd.DataFrame(columns=["address", "code"])
    return dict(auth=auth, transfers=tr, code=code)


def hop2_candidates(auth: pd.DataFrame, tr: pd.DataFrame, snd: pd.DataFrame, limit: int) -> list[str]:
    """Counterparties worth expanding: recipients of third-party-triggered outflows from
    authorities (candidate attacker sinks) and whoever funded the sponsors."""
    tr = tr.merge(snd, on="tx_hash", how="left")
    authorities, sponsors = set(auth.authority), set(auth.sponsor)
    third = tr[tr.src.isin(authorities) & (tr.sender != tr.src) & tr.sender.notna()]
    funders = tr[tr.dst.isin(sponsors)]
    counts = pd.concat([third.dst, funders.src]).value_counts()
    known = authorities | sponsors
    return [a for a in counts.index if a not in known][:limit]
