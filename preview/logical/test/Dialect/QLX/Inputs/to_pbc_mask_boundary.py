# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exercise QLX-to-PBC at the nonnegative-i64 Pauli-mask boundary."""

import sys

try:
    import _cudaq_logical_devpath
except ImportError:
    pass
from cudaq.mlir._mlir_libs import _qlxRuntime as runtime


def _module(support: int, *, rotation: bool) -> str:
    lines = [
        'module attributes {qlx.facets = [], qlx.ir_version = "0.3-draft", '
        'qlx.model_version = "0.3.9-proposed", qlx.profiles = ["p0"], '
        'qlx.stages = ["p0"]} {',
        '  qlx.program @mask_boundary : () -> i1 attributes '
        '{qlx.profile = "p0", qlx.stage = "p0"} {',
    ]
    current = []
    for index in range(support):
        name = f"%q{index}"
        lines.append(f'    {name} = qlx.prepare "zero" '
                     f'{{allocation = 0 : i64, value_index = {index} : i64}} '
                     ': !qlx.logical_qubit')
        current.append(name)

    # Z on the target conjugates through each preceding CX(target) to a Z
    # product over every operand. The same construction covers both the T
    # rotation column and the terminal Z measurement.
    target = support - 1
    for control in range(target):
        result = f"%cx{control}"
        lines.append(f"    {result}:2 = qlx.apply #qlx.action<cx>"
                     f"({current[control]}, {current[target]}) "
                     ": (!qlx.logical_qubit, !qlx.logical_qubit) -> "
                     "(!qlx.logical_qubit, !qlx.logical_qubit)")
        current[control] = f"{result}#0"
        current[target] = f"{result}#1"

    if rotation:
        lines.append(f"    %t = qlx.apply #qlx.action<t>({current[target]}) "
                     ": (!qlx.logical_qubit) -> !qlx.logical_qubit")
        current[target] = "%t"

    lines.extend((
        f"    %m = qlx.measure <Z> {current[target]} "
        ": !qlx.logical_qubit -> i1",
        "    qlx.return %m : i1",
        "  }",
        "}",
    ))
    return "\n".join(lines) + "\n"


def main(mode: str) -> None:
    if mode == "boundary":
        lowered = runtime.to_pbc(_module(63, rotation=True))
        maximum_mask = "z_mask = 9223372036854775807 : i64"
        if lowered.count(maximum_mask) < 2:
            raise AssertionError(
                "63-bit rotation and measurement masks were not preserved")
        print("63-bit rotation and measurement masks are representable")
    elif mode == "rotation-overflow":
        runtime.to_pbc(_module(64, rotation=True))
        raise AssertionError("64-bit rotation support was accepted")
    elif mode == "measurement-overflow":
        runtime.to_pbc(_module(64, rotation=False))
        raise AssertionError("64-bit measurement support was accepted")
    else:
        raise ValueError(f"unknown test mode: {mode}")


if __name__ == "__main__":
    main(sys.argv[1])
