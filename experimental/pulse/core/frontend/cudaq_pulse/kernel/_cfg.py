# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control flow graph builder operating on CanonicalInstr streams.

Entirely version-agnostic -- all version-specific knowledge is in
_bytecode_normalize.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ._bytecode_normalize import CanonicalInstr

_JUMP_OPS = frozenset({"JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE", "FOR_ITER"})


@dataclass
class BasicBlock:
    bid: int
    instrs: list[CanonicalInstr] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)


def build_cfg(instrs: list[CanonicalInstr]) -> dict[int, BasicBlock]:
    """Build a CFG from a canonical instruction stream.

    Returns a dict mapping block-id (= offset of the first instruction
    in the block) to BasicBlock.
    """
    if not instrs:
        return {}

    leaders: set[int] = {instrs[0].offset}
    offset_to_idx: dict[int, int] = {
        ci.offset: i for i, ci in enumerate(instrs)
    }

    for i, ci in enumerate(instrs):
        if ci.op in _JUMP_OPS:
            target = ci.arg
            if isinstance(target, int):
                leaders.add(target)
            if i + 1 < len(instrs):
                leaders.add(instrs[i + 1].offset)

    sorted_leaders = sorted(leaders)
    leader_to_bid: dict[int, int] = {off: off for off in sorted_leaders}

    blocks: dict[int, BasicBlock] = {}
    for idx, leader_off in enumerate(sorted_leaders):
        end_off = sorted_leaders[idx +
                                 1] if idx + 1 < len(sorted_leaders) else None
        start_i = offset_to_idx.get(leader_off)
        if start_i is None:
            blocks[leader_off] = BasicBlock(bid=leader_off)
            continue

        block_instrs: list[CanonicalInstr] = []
        for j in range(start_i, len(instrs)):
            if end_off is not None and instrs[j].offset >= end_off:
                break
            block_instrs.append(instrs[j])

        blocks[leader_off] = BasicBlock(bid=leader_off, instrs=block_instrs)

    for bid, block in blocks.items():
        if not block.instrs:
            continue
        last = block.instrs[-1]

        if last.op == "JUMP":
            target = last.arg
            if isinstance(target, int) and target in blocks:
                block.successors.append(target)
        elif last.op in ("JUMP_IF_FALSE", "JUMP_IF_TRUE"):
            target = last.arg
            fall = _fall_through(bid, sorted_leaders)
            if isinstance(target, int) and target in blocks:
                block.successors.append(target)
            if fall is not None and fall in blocks:
                block.successors.append(fall)
        elif last.op == "FOR_ITER":
            exit_target = last.arg
            fall = _fall_through(bid, sorted_leaders)
            if fall is not None and fall in blocks:
                block.successors.append(fall)
            if isinstance(exit_target, int) and exit_target in blocks:
                block.successors.append(exit_target)
        elif last.op == "RETURN":
            pass  # no successors
        else:
            fall = _fall_through(bid, sorted_leaders)
            if fall is not None and fall in blocks:
                block.successors.append(fall)

    for bid, block in blocks.items():
        for succ in block.successors:
            if succ in blocks:
                blocks[succ].predecessors.append(bid)

    return blocks


def _fall_through(bid: int, sorted_leaders: list[int]) -> Optional[int]:
    """Return the block-id of the fall-through successor, or None."""
    idx = sorted_leaders.index(bid)
    if idx + 1 < len(sorted_leaders):
        return sorted_leaders[idx + 1]
    return None
