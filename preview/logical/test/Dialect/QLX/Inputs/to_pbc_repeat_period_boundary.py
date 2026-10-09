# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Exercise bounded periodic-repeat normalization boundaries."""

import sys

try:
    import _cudaq_logical_devpath  # noqa: F401
except ImportError:
    pass

from cudaq.mlir._mlir_libs import _qlxRuntime as runtime


def _module_header() -> list[str]:
    return [
        'module attributes {qlx.facets = [], qlx.ir_version = "0.4-draft", '
        'qlx.model_version = "0.3.10-proposed", qlx.profiles = ["p0"], '
        'qlx.stages = ["p0"]} {',
    ]


def _module_start(name: str) -> list[str]:
    return _module_header() + [
        f'  qlx.program @{name} : () -> i1 attributes '
        '{qlx.profile = "p0", qlx.stage = "p0"} {',
    ]


def _period_overflow_module() -> str:
    # Disjoint 8- and 9-cycles have residual Clifford order 72, their least
    # common multiple.
    # Each adjacent SWAP is written as three supported CX operations.
    qubits = 17
    lines = _module_start("repeat_period_overflow")
    for index in range(qubits):
        lines.append(f'    %q{index} = qlx.prepare "zero" '
                     f'{{allocation = 0 : i64, value_index = {index} : i64}} '
                     ': !qlx.logical_qubit')
    arg_list = ", ".join(f"%a{index} : !qlx.logical_qubit = %q{index}"
                         for index in range(qubits))
    lines.extend(
        (f"    %r:{qubits} = cflow.repeat 100", f"        iter({arg_list}) {{"))
    current = [f"%a{index}" for index in range(qubits)]
    operation = 0

    def cx(control: int, target: int) -> None:
        nonlocal operation
        result = f"%cx{operation}"
        lines.append(f"      {result}:2 = qlx.apply #qlx.action<cx>"
                     f"({current[control]}, {current[target]}) "
                     ": (!qlx.logical_qubit, !qlx.logical_qubit) -> "
                     "(!qlx.logical_qubit, !qlx.logical_qubit)")
        current[control] = f"{result}#0"
        current[target] = f"{result}#1"
        operation += 1

    def swap(left: int, right: int) -> None:
        cx(left, right)
        cx(right, left)
        cx(left, right)

    for start, stop in ((0, 8), (8, 17)):
        for left in range(start, stop - 1):
            swap(left, left + 1)

    yielded = ", ".join(current)
    types = ", ".join("!qlx.logical_qubit" for _ in range(qubits))
    lines.extend((f"      cflow.yield {yielded} : {types}", "    }"))
    lines.append("    %m = qlx.measure #qlx.pauli<Z> %r#0 "
                 ": !qlx.logical_qubit -> i1")
    for index in range(1, qubits):
        lines.append(f"    qlx.discard %r#{index} : !qlx.logical_qubit")
    lines.extend(("    qlx.return %m : i1", "  }", "}"))
    return "\n".join(lines) + "\n"


def _clone_budget_module(constants: int) -> str:
    lines = _module_start("repeat_clone_budget")
    lines.extend((
        '    %q = qlx.prepare "zero" '
        '{allocation = 0 : i64, value_index = 0 : i64} '
        ': !qlx.logical_qubit',
        "    %r = cflow.repeat 2",
        "        iter(%arg : !qlx.logical_qubit = %q) {",
    ))
    for index in range(constants):
        lines.append(f"      %c{index} = arith.constant 0.0 : f64")
    lines.extend((
        "      %h = qlx.apply #qlx.action<h>(%arg) "
        ": (!qlx.logical_qubit) -> !qlx.logical_qubit",
        "      %t = qlx.apply #qlx.action<t>(%h) "
        ": (!qlx.logical_qubit) -> !qlx.logical_qubit",
        "      cflow.yield %t : !qlx.logical_qubit",
        "    }",
        "    %m = qlx.measure #qlx.pauli<Z> %r "
        ": !qlx.logical_qubit -> i1",
        "    qlx.return %m : i1",
        "  }",
        "}",
    ))
    return "\n".join(lines) + "\n"


def _direct_repeat_module(*, count: int, programs: int) -> str:
    lines = _module_header()
    for index in range(programs):
        lines.extend((
            f'  qlx.program @direct_{index} : () -> i1 attributes '
            '{qlx.profile = "p0", qlx.stage = "p0"} {',
            f'    %q{index} = qlx.prepare "zero" '
            f'{{allocation = 0 : i64, value_index = {index} : i64}} '
            ': !qlx.logical_qubit',
            f"    %r{index} = cflow.repeat {count}",
            f"        iter(%arg{index} : !qlx.logical_qubit = %q{index}) {{",
            f"      %h{index} = qlx.apply #qlx.action<h>(%arg{index}) "
            ": (!qlx.logical_qubit) -> !qlx.logical_qubit",
            f"      %t{index} = qlx.apply #qlx.action<t>(%h{index}) "
            ": (!qlx.logical_qubit) -> !qlx.logical_qubit",
            f"      %s{index} = qlx.apply #qlx.action<s>(%t{index}) "
            ": (!qlx.logical_qubit) -> !qlx.logical_qubit",
            f"      %tdg{index} = qlx.apply #qlx.action<tdg>(%s{index}) "
            ": (!qlx.logical_qubit) -> !qlx.logical_qubit",
            f"      cflow.yield %tdg{index} : !qlx.logical_qubit",
            "    }",
            f"    %m{index} = qlx.measure #qlx.pauli<Z> %r{index} "
            ": !qlx.logical_qubit -> i1",
            f"    qlx.return %m{index} : i1",
            "  }",
        ))
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(mode: str) -> None:
    if mode == "period-overflow":
        runtime.to_pbc(_period_overflow_module())
        raise AssertionError("residual period 72 was accepted")
    if mode == "clone-overflow":
        runtime.to_pbc(_clone_budget_module(2047))
        raise AssertionError("periodic clone budget overflow was accepted")
    if mode == "clone-boundary":
        runtime.to_pbc(_clone_budget_module(2046))
        print("exact 4096-operation periodic clone budget is accepted")
        return
    if mode == "direct-boundary":
        if not runtime.verify_clifford_t(_clone_budget_module(0)):
            raise AssertionError("cflow repeat failed Clifford+T verification")
        source = _direct_repeat_module(count=64, programs=1)
        lowered = runtime.to_pbc(source)
        if "cflow.repeat 1" not in lowered:
            raise AssertionError("count-64 direct phase chunk was not retained")
        if not runtime.verify_pbc(lowered):
            raise AssertionError("lowered cflow repeat failed PBC verification")
        print("exact 64-phase direct materialization limit is accepted")
        return
    if mode == "many-programs":
        programs = 64
        count = 32
        lowered = runtime.to_pbc(
            _direct_repeat_module(count=count, programs=programs))
        expected_rotations = programs * count * 2
        if lowered.count("#qlx.action<pauli_rotation>") != expected_rotations:
            raise AssertionError("many-program phase multiplicity changed")
        if not runtime.verify_pbc(lowered):
            raise AssertionError(
                "many-program lowering failed PBC verification")
        print("many direct-expanded programs preserve every SSA owner")
        return
    raise ValueError(f"unknown test mode: {mode}")


if __name__ == "__main__":
    main(sys.argv[1])
