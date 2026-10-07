"""DelegNet pipeline.

  python run.py collect  --chain ethereum          # scan -> code -> activity -> senders -> hop2
  python run.py label    --chain ethereum
  python run.py build    --chain ethereum          # temporal graph, motifs, code features
  python run.py pretrain                           # contrastive code encoder
  python run.py experiment --protocol temporal     # temporal | family | crosschain
  python run.py analyse                            # early warning, robustness, open-set, explanations
  python run.py report
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path

import pandas as pd

from delegnet.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run")


async def block_at_time(rpc, ts: int) -> int:
    """First block with timestamp >= ts (binary search)."""
    lo, hi = 0, int(await rpc.call("eth_blockNumber", []), 16)
    while lo < hi:
        mid = (lo + hi) // 2
        b = await rpc.call("eth_getBlockByNumber", [hex(mid), False])
        if int(b["timestamp"], 16) < ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


def sample_authorizations(cfg, chain, out, rng):
    """Stratified sample for activity collection: every delegation of small contracts, a random
    `cap` per large contract (uniform over time). All authorizations of a sampled authority are
    kept so its delegation history stays complete. Reads the full scan from `scan_root`."""
    from delegnet import collect as C
    cap = cfg["collect"]["per_contract_cap"]
    full = C.load_chain(Path(cfg["collect"]["scan_root"]) / chain)["auth"]
    tgt = full[~full.revoke]
    keep = tgt.groupby("delegate", group_keys=False).apply(
        lambda g: g if len(g) <= cap else g.sample(cap, random_state=0))
    s = full[full.authority.isin(set(keep.authority))]
    (out / "auth").mkdir(parents=True, exist_ok=True)
    s.to_parquet(out / "auth" / "sample.parquet", index=False)
    (out / "sample.json").write_text(json.dumps({"cap": cap, "full_authorizations": len(full),
                                                 "sampled_authorizations": len(s),
                                                 "contracts": int(full.delegate.nunique())}))
    log.info("%s: sampled %d of %d authorizations (cap %d per contract)", chain, len(s), len(full), cap)


async def collect(cfg, chain: str, steps: list[str]):
    from delegnet import collect as C

    cc, kc = cfg["chains"][chain], cfg["collect"]
    out = Path(cfg["data_root"]) / chain
    out.mkdir(parents=True, exist_ok=True)
    async with C.RPC(cc["endpoints"], concurrency=kc["concurrency"]) as rpc:
        rng_file = out / "range.json"
        if rng_file.exists():
            rng = json.loads(rng_file.read_text())
        else:
            rng = {"start": await block_at_time(rpc, cc["start_time_unix"]),
                   "end": await block_at_time(rpc, cc["end_time_unix"]) - 1}
            rng_file.write_text(json.dumps(rng))
        log.info("%s blocks %d-%d", chain, rng["start"], rng["end"])

        if "scan" in steps:
            await C.scan(rpc, out, cc["chain_id"], rng["start"], rng["end"], kc["scan_chunk"])
        if "sample" in steps:
            sample_authorizations(cfg, chain, out, rng)
        auth = C.load_chain(out)["auth"]
        if "code" in steps:
            await C.fetch_code(rpc, out, list(auth.delegate.unique()))
        tracked = sorted(set(auth.authority) | set(auth.sponsor))
        if "activity" in steps:
            await C.activity(rpc, out, tracked, rng["start"], rng["end"], "hop1",
                             kc["activity_block_chunk"], kc["activity_addr_chunk"], cc.get("token_logs", False))
            tr = C._read_dir(out / "transfers_hop1", C.TRANSFER_COLUMNS)
            await C.senders(rpc, out, list(tr.tx_hash.unique()))
        if "hop2" in steps:
            tr = C._read_dir(out / "transfers_hop1", C.TRANSFER_COLUMNS)
            snd = pd.read_parquet(out / "senders.parquet")
            extra = C.hop2_candidates(auth, tr, snd, kc["hop2_limit"])
            await C.fetch_code(rpc, out, extra)
            await C.activity(rpc, out, extra, rng["start"], rng["end"], "hop2",
                             kc["activity_block_chunk"], kc["activity_addr_chunk"], cc.get("token_logs", False))
            tr2 = C._read_dir(out / "transfers_hop2", C.TRANSFER_COLUMNS)
            await C.senders(rpc, out, list(tr2.tx_hash.unique()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--chain", default=None)
    ap.add_argument("--steps", default="scan,code,activity,hop2")
    ap.add_argument("--protocol", default="temporal")
    args = ap.parse_args()
    cfg = load_config(args.config)
    chains = [args.chain] if args.chain else list(cfg["chains"])

    if args.stage == "collect":
        for ch in chains:
            asyncio.run(collect(cfg, ch, args.steps.split(",")))
    elif args.stage == "label":
        from delegnet.labels import label_chain
        for ch in chains:
            label_chain(cfg, ch)
    elif args.stage == "build":
        from delegnet.graph import build_chain
        for ch in chains:
            build_chain(cfg, ch)
    elif args.stage == "pretrain":
        from delegnet.train import pretrain_code_encoder
        pretrain_code_encoder(cfg)
    elif args.stage == "experiment":
        from delegnet.experiments import run_protocol, run_zeroday
        if args.protocol == "zeroday":
            run_zeroday(cfg)
        else:
            run_protocol(cfg, args.protocol)
    elif args.stage == "analyse":
        from delegnet.experiments import run_analyses
        run_analyses(cfg)
    elif args.stage == "report":
        from delegnet.report import make_report
        make_report(cfg)
    else:
        raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    main()
