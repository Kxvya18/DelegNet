"""Bytecode pattern recognition.

`canonicalize` undoes the cheap obfuscations attackers use when they copy-paste a
sweeper (junk pushes, equivalent opcodes, padded trailers), so that the encoder and
the idiom detectors see the same thing for every variant of a family.
`idioms` turns the canonical form into interpretable indicators.
"""
from __future__ import annotations

import numpy as np

from ..evm import Instr, disassemble, mark_jump_refs, push_width, reachable

SELECTORS = {
    0xA9059CBB: "SEL_transfer", 0x23B872DD: "SEL_transferFrom", 0x095EA7B3: "SEL_approve",
    0xA22CB465: "SEL_setApprovalForAll", 0xD505ACCF: "SEL_permit", 0x70A08231: "SEL_balanceOf",
    0x2B67B570: "SEL_permit2", 0x1626BA7E: "SEL_isValidSignature", 0x19822F7C: "SEL_validateUserOp",
    0xB61D27F6: "SEL_execute", 0x47E1DA2A: "SEL_executeBatch", 0x34FCD5BE: "SEL_executeBatch2",
    0xE9AE5C53: "SEL_execute7579", 0x8DA5CB5B: "SEL_owner", 0x3659CFE6: "SEL_upgradeTo",
    0x4F1EF286: "SEL_upgradeToAndCall", 0x150B7A02: "SEL_onERC721Received", 0xF23A6E61: "SEL_onERC1155Received",
}
_NEUTRAL_ENV = {"GAS", "CALLVALUE", "PC", "MSIZE", "ADDRESS", "CALLER", "CALLDATASIZE", "RETURNDATASIZE"}
TERMINATORS = {"STOP", "RETURN", "REVERT", "INVALID", "SELFDESTRUCT", "JUMP"}


def _strip_junk(ins: list[Instr]) -> list[Instr]:
    out: list[Instr] = []
    for i in ins:
        out.append(i)
        while True:  # peephole: drop stack-neutral sequences ending in POP
            if len(out) >= 2 and out[-1].op == "POP" and (push_width(out[-2].op) or out[-2].op in _NEUTRAL_ENV
                                                         or out[-2].op == "PUSH0"):
                del out[-2:]
            elif len(out) >= 3 and out[-1].op == "POP" and out[-2].op == "ISZERO" and out[-3].op == "PUSH0":
                del out[-3:]
            elif (len(out) >= 4 and out[-1].op == "POP" and out[-2].op == "POP" and out[-3].op == "SWAP1"
                  and (push_width(out[-4].op) or out[-4].op == "PUSH0")
                  and len(out) >= 5 and (push_width(out[-5].op) or out[-5].op == "PUSH0")):
                del out[-5:]
            else:
                break
    return out


def canonicalize(code: bytes) -> list[Instr]:
    """Reachable code only, junk removed, equivalent forms unified, JUMPDEST markers dropped
    (their positions are what mutation shifts) and pushed code addresses tagged."""
    ins = reachable(mark_jump_refs(disassemble(code)))
    out: list[Instr] = []
    for i in ins:
        if i.op == "JUMPDEST":
            continue
        if i.op == "PUSH1" and i.arg == 0:
            i = Instr("PUSH0", None, i.offset)
        if out and out[-1].op == "ADDRESS" and i.op == "BALANCE":
            out[-1] = Instr("SELFBALANCE", None, out[-1].offset)
            continue
        if out and out[-1].op == "XOR" and i.op == "ISZERO":
            out[-1] = Instr("EQ", None, out[-1].offset)
            continue
        out.append(i)
    return _strip_junk(out)


def tokens(code: bytes) -> list[str]:
    """Token stream for the code encoder: opcodes plus semantic tags for well-known
    selectors, hard-coded addresses and precompile calls."""
    toks = []
    for i in canonicalize(code):
        if i.ref is not None or i.op in ("PUSH2", "PUSH3"):
            toks.append("PUSHTAG")   # code addresses (and 2-3 byte constants) shift under mutation
        elif i.op == "PUSH4" and i.arg in SELECTORS:
            toks.append(SELECTORS[i.arg])
        elif i.op == "PUSH20":
            toks.append("ADDR20")
        elif push_width(i.op) and i.arg is not None and 1 <= i.arg <= 10:
            toks.append(f"{i.op}_SMALL")  # precompile addresses, small constants
        else:
            toks.append(i.op)
    return toks


IDIOM_NAMES = [
    "forward_selfbalance_to_hardcoded", "forward_selfbalance", "caller_eq_hardcoded", "caller_eq_self",
    "origin_used", "ecrecover_call", "delegatecall", "delegatecall_to_calldata", "selfdestruct",
    "erc20_transfer", "erc20_transferFrom", "approve_or_setApprovalForAll", "permit",
    "upgrade_selector", "erc7579_or_4337", "n_selectors", "n_hardcoded_addrs", "n_calls",
    "n_sstore", "log_len", "fallback_has_call",
]


def idioms(code: bytes, window: int = 12) -> np.ndarray:
    ins = canonicalize(code)
    ops = [i.op for i in ins]
    n = len(ops)
    f = dict.fromkeys(IDIOM_NAMES, 0.0)
    if n == 0:
        return np.zeros(len(IDIOM_NAMES), dtype=np.float32)

    def near(k, names, before=window):
        return any(ops[j] in names for j in range(max(0, k - before), k))

    for k, i in enumerate(ins):
        op = i.op
        if op == "CALL":
            f["n_calls"] += 1
            if near(k, {"SELFBALANCE"}):
                f["forward_selfbalance"] = 1
                if near(k, {"PUSH20"}, 4):
                    f["forward_selfbalance_to_hardcoded"] = 1
        elif op == "EQ" and k >= 2:
            pair = {ops[k - 1], ops[k - 2]}
            if "CALLER" in pair and "PUSH20" in pair:
                f["caller_eq_hardcoded"] = 1
            if pair == {"CALLER", "ADDRESS"}:
                f["caller_eq_self"] = 1
        elif op == "ORIGIN":
            f["origin_used"] = 1
        elif op == "STATICCALL" and any(x.op.startswith("PUSH") and x.arg == 1 for x in ins[max(0, k - 4):k]):
            f["ecrecover_call"] = 1
        elif op == "DELEGATECALL":
            f["delegatecall"] = 1
            if near(k, {"CALLDATALOAD"}, 6) or near(k, {"SLOAD"}, 6):
                f["delegatecall_to_calldata"] = 1
        elif op == "SELFDESTRUCT":
            f["selfdestruct"] = 1
        elif op == "SSTORE":
            f["n_sstore"] += 1
        elif op == "PUSH20":
            f["n_hardcoded_addrs"] += 1
        elif op == "PUSH4":
            name = SELECTORS.get(i.arg, "")
            if k + 2 < n and "EQ" in ops[k + 1:k + 3]:
                f["n_selectors"] += 1
            f["erc20_transfer"] = max(f["erc20_transfer"], float(name == "SEL_transfer"))
            f["erc20_transferFrom"] = max(f["erc20_transferFrom"], float(name == "SEL_transferFrom"))
            f["approve_or_setApprovalForAll"] = max(f["approve_or_setApprovalForAll"], float(name in ("SEL_approve", "SEL_setApprovalForAll")))
            f["permit"] = max(f["permit"], float(name in ("SEL_permit", "SEL_permit2")))
            f["upgrade_selector"] = max(f["upgrade_selector"], float(name in ("SEL_upgradeTo", "SEL_upgradeToAndCall")))
            f["erc7579_or_4337"] = max(f["erc7579_or_4337"], float(name in ("SEL_execute7579", "SEL_validateUserOp")))
    # code reached when no selector matches: what follows the last dispatcher jump
    last_jump = max((k for k, o in enumerate(ops[: n // 2 + 1]) if o == "JUMPI"), default=-1)
    f["fallback_has_call"] = float(any(o in ("CALL", "DELEGATECALL") for o in ops[last_jump + 1:last_jump + 40]))
    f["log_len"] = np.log1p(n)  # canonical length, invariant to junk insertion
    for key in ("n_selectors", "n_hardcoded_addrs", "n_calls", "n_sstore"):
        f[key] = np.log1p(f[key])
    return np.array([float(f[k]) for k in IDIOM_NAMES], dtype=np.float32)
