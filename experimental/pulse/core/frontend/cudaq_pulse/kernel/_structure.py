# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured region recovery from a CFG.

Recovers ForLoop and IfElse regions from the flat CFG produced by
_cfg.py. Operates purely on canonical instructions and block structure --
no version-specific knowledge.

CPython's compiler produces reducible control flow for all constructs
we support (for/range, if/else). We use pattern matching rather than
general-purpose loop detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from ._bytecode_normalize import CanonicalInstr
from ._cfg import BasicBlock

# ── Region tree node types ───────────────────────────────────────────


@dataclass
class Block:
    """A basic block of straight-line instructions."""
    instrs: list[CanonicalInstr]


@dataclass
class ForLoop:
    """A for-range loop recovered from FOR_ITER..JUMP pattern."""
    loop_var: str
    range_setup: list[CanonicalInstr]  # LOAD range, LOAD N, CALL, GET_ITER
    body: list[Region]
    header_offset: int


@dataclass
class IfElse:
    """An if/else branch recovered from JUMP_IF_FALSE..JUMP pattern."""
    true_body: list[Region]
    false_body: list[Region]


Region = Union[Block, ForLoop, IfElse]

# ── Structure recovery ───────────────────────────────────────────────


def recover_structure(instrs: list[CanonicalInstr]) -> list[Region]:
    """Recover structured regions from a flat canonical instruction stream.

    Instead of building a full CFG, we do a single linear scan recognizing
    the bytecode patterns CPython emits for for-range loops and if/else.
    This is simpler and faster than CFG-based structure recovery for the
    restricted set of constructs we support.
    """
    return _recover(instrs, 0, len(instrs))


def _recover(instrs: list[CanonicalInstr], start: int,
             end: int) -> list[Region]:
    """Recursively recover regions in instrs[start:end]."""
    regions: list[Region] = []
    i = start

    while i < end:
        ci = instrs[i]

        if ci.op == "FOR_ITER":
            region, i = _recover_for_loop(instrs, i, end)
            regions.append(region)
            continue

        if ci.op in ("JUMP_IF_FALSE", "JUMP_IF_TRUE"):
            region, i = _recover_if_else(instrs, i, end)
            regions.append(region)
            continue

        straight: list[CanonicalInstr] = []
        while i < end:
            ci = instrs[i]
            if ci.op in ("FOR_ITER", "JUMP_IF_FALSE", "JUMP_IF_TRUE"):
                break
            straight.append(ci)
            i += 1

        if straight:
            regions.append(Block(instrs=straight))

    return regions


def _recover_for_loop(
    instrs: list[CanonicalInstr],
    for_iter_idx: int,
    end: int,
) -> tuple[ForLoop, int]:
    """Recover a ForLoop starting at the FOR_ITER instruction.

    Pattern (3.9):
        ... LOAD_GLOBAL(range) LOAD_CONST(N) CALL GET_ITER
        FOR_ITER(exit_offset)
        STORE_FAST(loop_var)
        <body>
        JUMP(back to FOR_ITER offset)
        <exit_offset>: ...
    """
    for_iter = instrs[for_iter_idx]
    exit_target = for_iter.arg
    header_offset = for_iter.offset
    loop_var = ""

    body_start = for_iter_idx + 1
    if body_start < end and instrs[body_start].op == "STORE_FAST":
        loop_var = instrs[body_start].arg
        body_start += 1

    body_end = for_iter_idx + 1
    for j in range(body_start, end):
        if instrs[j].op == "JUMP" and instrs[j].arg == header_offset:
            body_end = j
            break
    else:
        body_end = end

    body_instrs = instrs[body_start:body_end]
    body_regions = _recover(body_instrs, 0, len(body_instrs))

    range_setup: list[CanonicalInstr] = []
    setup_start = for_iter_idx
    for k in range(for_iter_idx - 1, -1, -1):
        if instrs[k].op == "GET_ITER":
            setup_start = k
            break
        if instrs[k].op in ("LOAD_GLOBAL", "LOAD_CONST", "CALL"):
            setup_start = k
        else:
            break

    next_idx = body_end + 1
    if next_idx < end and isinstance(exit_target, int):
        for j in range(next_idx, end):
            if instrs[j].offset >= exit_target:
                next_idx = j
                break

    return ForLoop(
        loop_var=loop_var,
        range_setup=instrs[setup_start:for_iter_idx],
        body=body_regions,
        header_offset=header_offset,
    ), next_idx


def _recover_if_else(
    instrs: list[CanonicalInstr],
    cond_idx: int,
    end: int,
) -> tuple[IfElse, int]:
    """Recover an IfElse starting at a JUMP_IF_FALSE instruction.

    Pattern:
        JUMP_IF_FALSE(else_or_end_offset)
        <true body>
        JUMP(end_offset)
        <else body>  (optional)
        <end_offset>: ...
    """
    cond = instrs[cond_idx]
    false_target = cond.arg
    is_negated = cond.op == "JUMP_IF_FALSE"

    true_start = cond_idx + 1
    true_end = true_start
    jump_end_idx: int | None = None

    for j in range(true_start, end):
        if isinstance(false_target, int) and instrs[j].offset >= false_target:
            true_end = j
            break
        if instrs[j].op == "JUMP" and j > true_start:
            true_end = j
            jump_end_idx = j
            break
    else:
        true_end = end

    true_instrs = instrs[true_start:true_end]

    if jump_end_idx is not None:
        jump_target = instrs[jump_end_idx].arg
        false_start = jump_end_idx + 1
        false_end = false_start
        for j in range(false_start, end):
            if isinstance(jump_target, int) and instrs[j].offset >= jump_target:
                false_end = j
                break
        else:
            false_end = end
        false_instrs = instrs[false_start:false_end]
        next_idx = false_end
    else:
        false_instrs = []
        next_idx = true_end

    true_regions = _recover(true_instrs, 0,
                            len(true_instrs)) if true_instrs else []
    false_regions = _recover(false_instrs, 0,
                             len(false_instrs)) if false_instrs else []

    if is_negated:
        return IfElse(true_body=true_regions,
                      false_body=false_regions), next_idx
    else:
        return IfElse(true_body=false_regions,
                      false_body=true_regions), next_idx
