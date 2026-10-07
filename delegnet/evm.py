"""Small EVM bytecode toolkit: assemble, disassemble, and semantics-preserving mutation.

The mutator is what we use both to create copy-paste variants of contracts in the
simulator and to attack the detectors in the robustness experiments.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

OPCODES = {
    0x00: "STOP", 0x01: "ADD", 0x02: "MUL", 0x03: "SUB", 0x04: "DIV", 0x05: "SDIV", 0x06: "MOD",
    0x07: "SMOD", 0x08: "ADDMOD", 0x09: "MULMOD", 0x0A: "EXP", 0x0B: "SIGNEXTEND",
    0x10: "LT", 0x11: "GT", 0x12: "SLT", 0x13: "SGT", 0x14: "EQ", 0x15: "ISZERO", 0x16: "AND",
    0x17: "OR", 0x18: "XOR", 0x19: "NOT", 0x1A: "BYTE", 0x1B: "SHL", 0x1C: "SHR", 0x1D: "SAR",
    0x20: "KECCAK256",
    0x30: "ADDRESS", 0x31: "BALANCE", 0x32: "ORIGIN", 0x33: "CALLER", 0x34: "CALLVALUE",
    0x35: "CALLDATALOAD", 0x36: "CALLDATASIZE", 0x37: "CALLDATACOPY", 0x38: "CODESIZE",
    0x39: "CODECOPY", 0x3A: "GASPRICE", 0x3B: "EXTCODESIZE", 0x3C: "EXTCODECOPY",
    0x3D: "RETURNDATASIZE", 0x3E: "RETURNDATACOPY", 0x3F: "EXTCODEHASH",
    0x40: "BLOCKHASH", 0x41: "COINBASE", 0x42: "TIMESTAMP", 0x43: "NUMBER", 0x44: "PREVRANDAO",
    0x45: "GASLIMIT", 0x46: "CHAINID", 0x47: "SELFBALANCE", 0x48: "BASEFEE", 0x49: "BLOBHASH",
    0x4A: "BLOBBASEFEE",
    0x50: "POP", 0x51: "MLOAD", 0x52: "MSTORE", 0x53: "MSTORE8", 0x54: "SLOAD", 0x55: "SSTORE",
    0x56: "JUMP", 0x57: "JUMPI", 0x58: "PC", 0x59: "MSIZE", 0x5A: "GAS", 0x5B: "JUMPDEST",
    0x5C: "TLOAD", 0x5D: "TSTORE", 0x5E: "MCOPY", 0x5F: "PUSH0",
    0xF0: "CREATE", 0xF1: "CALL", 0xF2: "CALLCODE", 0xF3: "RETURN", 0xF4: "DELEGATECALL",
    0xF5: "CREATE2", 0xFA: "STATICCALL", 0xFD: "REVERT", 0xFE: "INVALID", 0xFF: "SELFDESTRUCT",
}
for _i in range(32):
    OPCODES[0x60 + _i] = f"PUSH{_i + 1}"
for _i in range(16):
    OPCODES[0x80 + _i] = f"DUP{_i + 1}"
    OPCODES[0x90 + _i] = f"SWAP{_i + 1}"
for _i in range(5):
    OPCODES[0xA0 + _i] = f"LOG{_i}"
NAME_TO_BYTE = {v: k for k, v in OPCODES.items()}
OPCODE_NAMES = sorted(NAME_TO_BYTE, key=NAME_TO_BYTE.get)


def push_width(op: str) -> int:
    return int(op[4:]) if op.startswith("PUSH") and op != "PUSH0" else 0


@dataclass
class Instr:
    op: str
    arg: int | None = None
    offset: int = -1        # offset in the original code, -1 for inserted instructions
    ref: int | None = None  # for "PUSH dest; JUMP(I)" pairs: original offset of the target


def assemble(tokens: list[str]) -> bytes:
    """Assemble tokens like 'PUSH20 0xab..', 'PUSH2 @done' and ':done' (label, emits JUMPDEST)."""
    items = []
    for tok in tokens:
        tok = tok.strip()
        if tok.startswith(":"):
            items.append(("LABEL", tok[1:]))
            continue
        parts = tok.split()
        op = parts[0].upper()
        if op not in NAME_TO_BYTE:
            raise ValueError(f"unknown opcode {op}")
        arg = None
        if len(parts) > 1:
            arg = parts[1][1:] if parts[1].startswith("@") else int(parts[1], 0)
        items.append((op, arg))

    labels, off = {}, 0
    for op, arg in items:
        if op == "LABEL":
            labels[arg] = off
            off += 1
        else:
            off += 1 + push_width(op)

    out = bytearray()
    for op, arg in items:
        if op == "LABEL":
            out.append(0x5B)
            continue
        out.append(NAME_TO_BYTE[op])
        w = push_width(op)
        if w:
            val = labels[arg] if isinstance(arg, str) else (arg or 0)
            out += val.to_bytes(w, "big")
    return bytes(out)


def disassemble(code: bytes) -> list[Instr]:
    out, i = [], 0
    while i < len(code):
        op = OPCODES.get(code[i], "INVALID")
        w = push_width(op)
        arg = int.from_bytes(code[i + 1:i + 1 + w].ljust(w, b"\x00"), "big") if w else None
        if op == "INVALID":
            arg = code[i]  # keep the raw byte of undefined opcodes so layout is lossless
        out.append(Instr(op, arg, i))
        i += 1 + w
    return out


def opcode_sequence(code: bytes) -> list[str]:
    return [ins.op for ins in disassemble(code)]


# ---------------------------------------------------------------- mutation

def _neutral(rng: random.Random) -> list[Instr]:
    """Stack-neutral junk snippets."""
    choice = rng.randrange(5)
    if choice == 0:
        return [Instr("JUMPDEST")]
    if choice == 1:
        return [Instr("PUSH1", rng.randrange(256)), Instr("POP")]
    if choice == 2:
        return [Instr(rng.choice(["GAS", "CALLVALUE", "PC", "MSIZE"])), Instr("POP")]
    if choice == 3:
        return [Instr("PUSH2", rng.randrange(65536)), Instr("PUSH1", rng.randrange(256)),
                Instr("SWAP1"), Instr("POP"), Instr("POP")]
    return [Instr("PUSH0"), Instr("ISZERO"), Instr("POP")]


def _substitute(ins: Instr, rng: random.Random, p: float) -> list[Instr]:
    if rng.random() >= p:
        return [ins]
    if ins.op == "SELFBALANCE":
        return [Instr("ADDRESS"), Instr("BALANCE")]
    if ins.op == "EQ":                     # a == b  <=>  (a ^ b) == 0
        return [Instr("XOR"), Instr("ISZERO")]
    if ins.op == "PUSH0":
        return [Instr("PUSH1", 0)]
    if ins.op == "PUSH1" and ins.arg == 0 and ins.ref is None:
        return [Instr("PUSH0")]
    return [ins]


def _layout(ins: list[Instr]) -> bytes:
    widths = [push_width(i.op) for i in ins]
    for _ in range(4):  # widen jump pushes until targets fit
        pos, newpos = 0, {}
        for i, w in zip(ins, widths):
            if i.op == "JUMPDEST" and i.offset >= 0:
                newpos[i.offset] = pos
            pos += 1 + w
        grew = False
        for k, i in enumerate(ins):
            if i.ref is not None:
                need = max(1, (newpos[i.ref].bit_length() + 7) // 8)
                if need > widths[k]:
                    widths[k], grew = need, True
        if not grew:
            break
    out = bytearray()
    for i, w in zip(ins, widths):
        op = f"PUSH{w}" if push_width(i.op) else i.op
        out.append(i.arg if op == "INVALID" and i.arg is not None else NAME_TO_BYTE[op])
        if w:
            val = newpos[i.ref] if i.ref is not None else (i.arg or 0)
            out += val.to_bytes(w, "big")
    return bytes(out)


def mark_jump_refs(ins: list[Instr]) -> list[Instr]:
    """Tag pushes that carry a code address: static `PUSH dest; JUMP(I)` pairs and the
    dynamic return addresses solc pushes for internal calls (any PUSH2+ equal to a
    JUMPDEST offset). This is the usual heuristic of EVM bytecode rewriters."""
    dests = {i.offset for i in ins if i.op == "JUMPDEST"}
    for k, i in enumerate(ins):
        w = push_width(i.op)
        if not w or i.arg not in dests:
            continue
        static = k + 1 < len(ins) and ins[k + 1].op in ("JUMP", "JUMPI")
        if w >= 2 or static or i.arg not in _COMMON_PUSH1:
            i.ref = i.arg
    return ins


# PUSH1 values that are almost always memory/calldata offsets or masks, never return addresses
_COMMON_PUSH1 = set(range(0, 0x10)) | {0x1F, 0xFF, 0xE0} | set(range(0x20, 0x100, 0x20)) | \
    set(range(0x04, 0x100, 0x20))


def reachable(ins: list[Instr]) -> list[Instr]:
    """Instructions reachable from entry, following fall-through and every JUMPDEST whose
    offset is pushed somewhere reachable. Drops metadata, data sections and junk tails."""
    if not ins:
        return ins
    at = {i.offset: k for k, i in enumerate(ins)}
    seen, todo, pushed = set(), [0], set()
    while todo:
        k = todo.pop()
        while k < len(ins) and k not in seen:
            seen.add(k)
            i = ins[k]
            if push_width(i.op) and i.arg is not None:
                tgt = at.get(i.arg)
                if tgt is not None and ins[tgt].op == "JUMPDEST" and i.arg not in pushed:
                    pushed.add(i.arg)
                    todo.append(tgt)
            if i.op in ("STOP", "RETURN", "REVERT", "INVALID", "SELFDESTRUCT", "JUMP"):
                break
            k += 1
    return [ins[k] for k in sorted(seen)]


def mutate_bytecode(code: bytes, rng: random.Random, strength: float = 1.0) -> bytes:
    """Return a functionally equivalent variant of `code` (junk insertion, equivalent
    opcode substitution, unreachable trailer); every pushed code address is re-linked."""
    if strength <= 0:
        return code
    base = len([i for i in reachable(mark_jump_refs(disassemble(code))) if i.op != "JUMPDEST"])
    for attempt in range(6):
        new = _mutate_once(code, rng, strength if attempt < 5 else 0.0)
        got = [i for i in reachable(mark_jump_refs(disassemble(new))) if i.op != "JUMPDEST"]
        if len(got) >= base:   # control flow survived (junk only ever adds instructions)
            return new
    return code


def _mutate_once(code: bytes, rng: random.Random, strength: float) -> bytes:
    ins = mark_jump_refs(disassemble(code))
    p_ins, p_sub = min(0.5, 0.06 * strength), min(0.9, 0.25 * max(strength, 1.0))
    out = []
    for i in ins:
        if i.op not in ("JUMP", "JUMPI") and rng.random() < p_ins:
            out.extend(_neutral(rng))
        out.extend(_substitute(replace(i), rng, p_sub))
    new = _layout(out)
    tail = rng.randint(0, int(24 * strength))
    if tail:  # unreachable trailer, like solc metadata
        new += b"\xfe" + bytes(rng.randrange(256) for _ in range(tail))
    return new
