# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q Quake to canonical CUDA-Q Logical P0 import tests."""

import importlib
import re
from pathlib import Path

import pytest

import cudaq.logical
import cudaq.mlir.ir as mlir_ir
from cudaq.logical._native import native

pytestmark = pytest.mark.skipif(
    not native.has_quake_import,
    reason="CUDA-Q Logical was built without CUDA-Q Quake support",
)

WIRE_BELL = r"""
module attributes {cc.python_uniqued = "bell..0xabc"} {
  func.func @__nvqpp__mlirgen__bell..0xabc() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.h %0 : (!quake.wire) -> !quake.wire
    %3:2 = quake.x [%2] %1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    %4 = quake.t %3#0 : (!quake.wire) -> !quake.wire
    %m0, %w0 = quake.mz %4 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %m1, %w1 = quake.mz %3#1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    quake.sink %w0 : !quake.wire
    quake.sink %w1 : !quake.wire
    return
  }
  func.func private @malloc(i64) -> !cc.ptr<i8>
  llvm.func @cudaqRegisterLambdaName(!llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
}
"""

MIXED_ENTRY_FAILURE = r"""
module {
  func.func @__nvqpp__mlirgen__good() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    quake.sink %0 : !quake.wire
    return
  }
  func.func @__nvqpp__mlirgen__bad() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.alloca !quake.ref
    quake.h %0 : (!quake.ref) -> ()
    return
  }
}
"""

ENTRY_OWNER_LEAK = r"""
module {
  func.func @__nvqpp__mlirgen__leak()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %wire = quake.null_wire
    return
  }
}
"""

DEAD_CARRY_THEN_FAILURE = r"""
module {
  func.func @__nvqpp__mlirgen__dead_carry_then_failure()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %wire = quake.null_wire
    %dead = cc.undef i64
    %loop:3 = cc.loop while (
        (%iv = %c0, %unused = %dead, %iter = %wire)
        -> (i64, i64, !quake.wire)) {
      %condition = arith.cmpi ne, %iv, %c2 : i64
      cc.condition %condition(
          %iv, %unused, %iter : i64, i64, !quake.wire)
    } do {
    ^bb0(%iv: i64, %unused: i64, %iter: !quake.wire):
      %next_wire = quake.h %iter : (!quake.wire) -> !quake.wire
      cc.continue %iv, %unused, %next_wire : i64, i64, !quake.wire
    } step {
    ^bb0(%iv: i64, %unused: i64, %iter: !quake.wire):
      %next = arith.addi %iv, %c1 : i64
      cc.continue %next, %unused, %iter : i64, i64, !quake.wire
    }
    quake.sink %loop#2 : !quake.wire
    %unsupported = quake.alloca !quake.ref
    quake.h %unsupported : (!quake.ref) -> ()
    return
  }
}
"""

TOP_LEVEL_EXECUTABLE_QLX = r"""
module {
  func.func @__nvqpp__mlirgen__good() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    quake.sink %0 : !quake.wire
    return
  }
  %orphan = qlx.prepare "zero" {allocation = 99 : i64, value_index = 0 : i64}
      : !qlx.logical_qubit
  qlx.discard %orphan : !qlx.logical_qubit
}
"""

SAME_BASE_VARIANTS = r"""
module {
  func.func @__nvqpp__mlirgen__foo..0x1() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    quake.sink %0 : !quake.wire
    return
  }
  func.func @__nvqpp__mlirgen__foo..0x2() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.h %0 : (!quake.wire) -> !quake.wire
    quake.sink %1 : !quake.wire
    return
  }
}
"""


def test_live_module_pass_has_no_file_or_text_round_trip():
    # Load the plugin before creating the context so Quake parses as a
    # registered dialect. A live module from SDK-mode CUDA-Q already has that
    # dialect loaded from the same shared compiler library.
    cudaq.logical.compiler.import_quake(WIRE_BELL)
    module = mlir_ir.Module.parse(WIRE_BELL, mlir_ir.Context())

    converted = cudaq.logical.compiler.convert_quake_to_p0(module)

    assert converted is module
    assert "qlx.program @bell" in str(module)
    assert "quake." not in str(module)


def test_closed_import_and_public_conversion_each_verify_linearity_once(
    monkeypatch,):
    quake_import = importlib.import_module("cudaq.logical.compiler.quake")
    original = quake_import.verify_linearity
    subjects = []

    def counted(module, *, subject=None):
        subjects.append(subject)
        return original(module, subject=subject)

    monkeypatch.setattr(quake_import, "verify_linearity", counted)
    cudaq.logical.compiler.import_quake(WIRE_BELL)
    assert subjects == ["imported Quake program @bell"]

    module = mlir_ir.Module.parse(WIRE_BELL, mlir_ir.Context())
    cudaq.logical.compiler.convert_quake_to_p0(module)
    assert subjects[-1] == "converted Quake P0 module"
    assert len(subjects) == 2


def test_live_cudaq_module_uses_the_same_in_process_pass():
    from cudaq.mlir import ir as cudaq_ir
    from cudaq.mlir.dialects import quake

    context = cudaq_ir.Context()
    quake.register_dialect(context=context)
    with context:
        module = cudaq_ir.Module.parse(WIRE_BELL)

    converted = cudaq.logical.compiler.convert_quake_to_p0(module)
    # The pass mutates and returns the caller's exact live module object.
    assert converted is module
    assert "qlx.program @bell" in str(module)
    assert "quake." not in str(module)


def test_live_module_conversion_rolls_back_when_a_later_entry_fails():
    cudaq.logical.compiler.import_quake(WIRE_BELL)
    module = mlir_ir.Module.parse(MIXED_ENTRY_FAILURE, mlir_ir.Context())
    before = str(module)

    with pytest.raises(RuntimeError, match="value-semantics"):
        cudaq.logical.compiler.convert_quake_to_p0(module)

    assert str(module) == before
    assert "qlx.program" not in str(module)


def test_live_module_conversion_rolls_back_when_owner_closure_fails():
    cudaq.logical.compiler.import_quake(WIRE_BELL)
    module = mlir_ir.Module.parse(ENTRY_OWNER_LEAK, mlir_ir.Context())
    before = str(module)

    with pytest.raises(RuntimeError, match="leaves 1 live quantum owner"):
        cudaq.logical.compiler.convert_quake_to_p0(module)

    assert str(module) == before
    assert "qlx.program" not in str(module)


def test_live_module_conversion_rolls_back_boundary_cleanup_on_failure():
    cudaq.logical.compiler.import_quake(WIRE_BELL)
    module = mlir_ir.Module.parse(DEAD_CARRY_THEN_FAILURE, mlir_ir.Context())
    before = str(module)

    with pytest.raises(RuntimeError, match="value-semantics"):
        cudaq.logical.compiler.convert_quake_to_p0(module)

    assert str(module) == before
    assert "cc.undef" in str(module)
    assert "qlx.program" not in str(module)


def test_quake_import_rejects_orphan_top_level_qlx_execution():
    with pytest.raises(RuntimeError, match="unexpected top-level operation"):
        cudaq.logical.compiler.import_quake(TOP_LEVEL_EXECUTABLE_QLX)


def test_quake_import_selects_exact_source_entry_before_name_normalization():
    p0 = cudaq.logical.compiler.import_quake(SAME_BASE_VARIANTS,
                                             root="foo..0x2")
    assert p0.root.symbol == "foo"
    assert cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL).actions == {
            "qlx_standard_h": 1
        }


def test_quake_import_rejects_ambiguous_abbreviated_source_entry():
    with pytest.raises(RuntimeError, match="identified 2 source entries"):
        cudaq.logical.compiler.import_quake(SAME_BASE_VARIANTS, root="foo")


REFERENCE_QUBIT = r"""
module {
  func.func @__nvqpp__mlirgen__reference() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.alloca !quake.ref
    quake.h %0 : (!quake.ref) -> ()
    return
  }
}
"""

WIRE_H = r"""
module {
  func.func @__nvqpp__mlirgen__logical_h() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.h %0 : (!quake.wire) -> !quake.wire
    quake.sink %1 : !quake.wire
    return
  }
}
"""

WIRE_CCX = r"""
module {
  func.func @__nvqpp__mlirgen__ccx() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %3:3 = quake.x [%0, %1] %2 : (!quake.wire, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire, !quake.wire)
    quake.sink %3#0 : !quake.wire
    quake.sink %3#1 : !quake.wire
    quake.sink %3#2 : !quake.wire
    return
  }
}
"""

WIRE_TDG = r"""
module {
  func.func @__nvqpp__mlirgen__tdg() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.t<adj> %0 : (!quake.wire) -> !quake.wire
    quake.sink %1 : !quake.wire
    return
  }
}
"""

WIRE_FREDKIN = r"""
module {
  func.func @__nvqpp__mlirgen__fredkin() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %3:3 = quake.swap [%0] %1, %2 : (!quake.wire, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire, !quake.wire)
    quake.sink %3#0 : !quake.wire
    quake.sink %3#1 : !quake.wire
    quake.sink %3#2 : !quake.wire
    return
  }
}
"""

NEGATED_CONTROL = r"""
module {
  func.func @__nvqpp__mlirgen__negated_control() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2:2 = quake.x [%0 neg [true]] %1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    quake.sink %2#0 : !quake.wire
    quake.sink %2#1 : !quake.wire
    return
  }
}
"""

EXCESSIVE_CONTROLS = r"""
module {
  func.func @__nvqpp__mlirgen__excessive_controls() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %3 = quake.null_wire
    %4:4 = quake.x [%0, %1, %2] %3 : (!quake.wire, !quake.wire, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire, !quake.wire, !quake.wire)
    quake.sink %4#0 : !quake.wire
    quake.sink %4#1 : !quake.wire
    quake.sink %4#2 : !quake.wire
    quake.sink %4#3 : !quake.wire
    return
  }
}
"""

EXCESSIVE_SWAP_TARGETS = r"""
module {
  func.func @__nvqpp__mlirgen__excessive_swap_targets() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %3:3 = quake.swap %0, %1, %2 : (!quake.wire, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire, !quake.wire)
    quake.sink %3#0 : !quake.wire
    quake.sink %3#1 : !quake.wire
    quake.sink %3#2 : !quake.wire
    return
  }
}
"""

QUANTUM_ENTRY_ARGUMENT = r"""
module {
  func.func @__nvqpp__mlirgen__quantum_argument(%arg0: !quake.wire) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.h %arg0 : (!quake.wire) -> !quake.wire
    quake.sink %0 : !quake.wire
    return
  }
}
"""

QUANTUM_CFG_BLOCK_ARGUMENT = r"""
module {
  func.func @__nvqpp__mlirgen__quantum_block_argument() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    cf.br ^bb1(%0 : !quake.wire)
  ^bb1(%arg0: !quake.wire):
    %1 = quake.h %arg0 : (!quake.wire) -> !quake.wire
    quake.sink %1 : !quake.wire
    return
  }
}
"""

DYNAMIC_ROTATION_ARGUMENT = r"""
module {
  func.func @__nvqpp__mlirgen__dynamic_rotation(%theta: f64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.rz (%theta) %0 : (f64, !quake.wire) -> !quake.wire
    quake.sink %1 : !quake.wire
    return
  }
}
"""

NESTED_ADAPTIVE_CONTROL = r"""
module {
  func.func @__nvqpp__mlirgen__nested_adaptive() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %m, %w = quake.mz %0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %c = quake.discriminate %m : (!cc.measure_handle) -> i1
    %r = cc.if (%c) ((%a = %1)) -> (!quake.wire) {
      %inner = cc.if (%c) ((%b = %a)) -> (!quake.wire) {
        %x = quake.x %b : (!quake.wire) -> !quake.wire
        cc.continue %x : !quake.wire
      } else {
        cc.continue %b : !quake.wire
      }
      cc.continue %inner : !quake.wire
    } else {
      cc.continue %a : !quake.wire
    }
    quake.sink %w : !quake.wire
    quake.sink %r : !quake.wire
    return %c : i1
  }
}
"""

UNSUPPORTED_PHASED_RX = r"""
module {
  func.func @__nvqpp__mlirgen__phased_rx() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %theta = arith.constant 0.25 : f64
    %phase = arith.constant 0.5 : f64
    %0 = quake.null_wire
    %1 = quake.phased_rx (%theta, %phase) %0 : (f64, f64, !quake.wire) -> !quake.wire
    quake.sink %1 : !quake.wire
    return
  }
}
"""


def test_wire_semantics_quake_imports_as_verified_p0():
    p0 = cudaq.logical.compiler.import_quake(WIRE_BELL)

    assert p0.stage == cudaq.logical.stages.P0
    assert p0.root.symbol == "bell"
    assert tuple(group.name for group in p0.values) == ("alloc0", "alloc1")
    assert [record.result for record in p0.evidence] == ["pass", "pass"]

    logical = cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL)
    assert logical.logical_qubits_peak == 2
    assert logical.actions == {
        "qlx_standard_h": 1,
        "qlx_standard_cx": 1,
        "qlx_standard_t": 1,
    }
    assert logical.synthesis_demand == {"qlx_standard_t": 1}


def test_quake_path_import_produces_p0(tmp_path: Path):
    source = tmp_path / "bell.qke"
    source.write_text(WIRE_BELL)

    p0 = cudaq.logical.compiler.import_quake(source)
    assert p0.stage == cudaq.logical.stages.P0
    assert p0.root.symbol == "bell"


def test_quake_import_two_control_x_has_logical_ccz_demand():
    p0 = cudaq.logical.compiler.import_quake(WIRE_CCX)
    logical = cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL)
    assert logical.synthesis_demand == {"qlx_standard_ccz": 1}


def test_quake_import_tdg_remains_a_standard_action():
    p0 = cudaq.logical.compiler.import_quake(WIRE_TDG)
    logical = cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL)
    assert logical.actions == {"qlx_standard_tdg": 1}
    assert logical.synthesis_demand == {"qlx_standard_tdg": 1}


def test_quake_import_rejects_controlled_swap():
    with pytest.raises(RuntimeError, match="controlled swap") as excinfo:
        cudaq.logical.compiler.import_quake(WIRE_FREDKIN)
    # The captured diagnostic carries a source location (line:column).
    assert re.search(r":\d+:\d+", str(excinfo.value))


def test_quake_import_rejects_negated_control():
    with pytest.raises(RuntimeError, match="negated controls"):
        cudaq.logical.compiler.import_quake(NEGATED_CONTROL)


def test_quake_import_rejects_excessive_controls():
    with pytest.raises(RuntimeError, match="unsupported gate shape"):
        cudaq.logical.compiler.import_quake(EXCESSIVE_CONTROLS)


def test_quake_import_rejects_excessive_swap_targets():
    # The Quake op verifier rejects this malformed shape before the conversion
    # pass runs; it is still a fail-closed boundary with an actionable error.
    with pytest.raises(cudaq.mlir.ir.MLIRError,
                       match="number of targets is equal to 2"):
        cudaq.logical.compiler.import_quake(EXCESSIVE_SWAP_TARGETS)


def test_quake_import_rejects_quantum_entry_argument():
    with pytest.raises(
            RuntimeError,
            match="specialized entry point with no quantum arguments"):
        cudaq.logical.compiler.import_quake(QUANTUM_ENTRY_ARGUMENT)


def test_quake_import_rejects_quantum_cfg_block_argument():
    with pytest.raises(RuntimeError, match="quantum CFG block arguments"):
        cudaq.logical.compiler.import_quake(QUANTUM_CFG_BLOCK_ARGUMENT)


def test_quake_import_rejects_unspecialized_dynamic_angle():
    with pytest.raises(RuntimeError,
                       match="classical argument to be constant-folded"):
        cudaq.logical.compiler.import_quake(DYNAMIC_ROTATION_ARGUMENT)


def test_quake_import_rejects_unsupported_gate():
    with pytest.raises(
            RuntimeError,
            match="outside the typed Quake-to-P0 conversion contract"):
        cudaq.logical.compiler.import_quake(UNSUPPORTED_PHASED_RX)


def test_quake_import_rejects_nested_adaptive_control():
    with pytest.raises(RuntimeError,
                       match="flatten or outline the nested region"):
        cudaq.logical.compiler.import_quake(NESTED_ADAPTIVE_CONTROL)


def test_quake_import_rejects_reference_semantics():
    with pytest.raises(RuntimeError, match="only value-semantics !quake.wire"):
        cudaq.logical.compiler.import_quake(REFERENCE_QUBIT)


# --- Phase B: folded loops + conditionals (normalized cc.loop / cc.if) -------


# A normalized (post `cudaq-opt --cc-loop-normalize`) constant-trip loop:
# `arith.cmpi ne` / initial value 0 / step +1, wrapped in cc.scope, carrying two
# wires.
def _normalized_loop(trip: int, predicate: str = "ne") -> str:
    return f"""
module {{
  func.func @__nvqpp__mlirgen__loopy() attributes {{"cudaq-entrypoint", "cudaq-kernel"}} {{
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %ct = arith.constant {trip} : i32
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2:2 = cc.scope -> (!quake.wire, !quake.wire) {{
      %9:3 = cc.loop while ((%a0 = %0, %a1 = %1, %iv = %c0) -> (!quake.wire, !quake.wire, i32)) {{
        %c = arith.cmpi {predicate}, %iv, %ct : i32
        cc.condition %c(%a0, %a1, %iv : !quake.wire, !quake.wire, i32)
      }} do {{
      ^bb0(%a0: !quake.wire, %a1: !quake.wire, %iv: i32):
        %h = quake.h %a0 : (!quake.wire) -> !quake.wire
        %x:2 = quake.x [%h] %a1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        cc.continue %x#0, %x#1, %iv : !quake.wire, !quake.wire, i32
      }} step {{
      ^bb0(%a0: !quake.wire, %a1: !quake.wire, %iv: i32):
        %n = arith.addi %iv, %c1 : i32
        cc.continue %a0, %a1, %n : !quake.wire, !quake.wire, i32
      }}
      cc.continue %9#0, %9#1 : !quake.wire, !quake.wire
    }}
    quake.sink %2#0 : !quake.wire
    quake.sink %2#1 : !quake.wire
    return
  }}
}}
"""


def _normalized_t_loop(trip: int) -> str:
    return _normalized_loop(trip).replace(
        """        %h = quake.h %a0 : (!quake.wire) -> !quake.wire
        %x:2 = quake.x [%h] %a1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        cc.continue %x#0, %x#1, %iv : !quake.wire, !quake.wire, i32""",
        """        %t = quake.t %a0 : (!quake.wire) -> !quake.wire
        cc.continue %t, %a1, %iv : !quake.wire, !quake.wire, i32""",
    )


MEAS_IF = r"""
module {
  func.func @__nvqpp__mlirgen__cond() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %m, %w = quake.mz %0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %c = quake.discriminate %m : (!cc.measure_handle) -> i1
    %r = cc.if (%c) ((%a = %1)) -> (!quake.wire) {
      %x = quake.x %a : (!quake.wire) -> !quake.wire
      cc.continue %x : !quake.wire
    } else {
      cc.continue %a : !quake.wire
    }
    quake.sink %w : !quake.wire
    quake.sink %r : !quake.wire
    return %c : i1
  }
}
"""

RESOURCE_MEAS_IF = MEAS_IF.replace(
    "%x = quake.x %a : (!quake.wire) -> !quake.wire",
    "%x = quake.t %a : (!quake.wire) -> !quake.wire",
)


def test_quake_import_folded_loop_multiplies_logical_estimate():
    p0 = cudaq.logical.compiler.import_quake(_normalized_loop(5))
    assert cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL).actions == {
            "qlx_standard_h": 5,
            "qlx_standard_cx": 5,
        }


def test_quake_import_zero_trip_loop_preserves_carries():
    p0 = cudaq.logical.compiler.import_quake(_normalized_loop(0))
    actions = cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL).actions
    assert actions.get("qlx_standard_h", 0) == 0


def test_quake_import_standard_t_stays_inside_folded_loop():
    p0 = cudaq.logical.compiler.import_quake(_normalized_t_loop(5))
    logical = cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL)
    assert logical.synthesis_demand == {"qlx_standard_t": 5}


def test_quake_import_rejects_unnormalized_loop():
    with pytest.raises(RuntimeError, match="condition must use"):
        cudaq.logical.compiler.import_quake(_normalized_loop(5,
                                                             predicate="sle"))


def test_quake_import_measurement_conditional_remains_p0():
    p0 = cudaq.logical.compiler.import_quake(MEAS_IF)
    assert p0.stage == cudaq.logical.stages.P0
    # Both branches are counted (conservative static upper bound).
    assert cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL).actions == {
            "qlx_standard_x": 1
        }


def test_quake_import_standard_t_stays_inside_adaptive_branch():
    p0 = cudaq.logical.compiler.import_quake(RESOURCE_MEAS_IF)
    logical = cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL)
    assert logical.synthesis_demand == {"qlx_standard_t": 1}


# --- rotations: R1/RX/RY/RZ -> ``qlx.apply<pauli_rotation>`` ----------------

ROTATIONS = r"""
module {
  func.func @__nvqpp__mlirgen__rots() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.5 : f64
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %rx = quake.rx (%a) %0 : (f64, !quake.wire) -> !quake.wire
    %ry = quake.ry (%a) %1 : (f64, !quake.wire) -> !quake.wire
    %rz = quake.rz (%a) %2 : (f64, !quake.wire) -> !quake.wire
    quake.sink %rx : !quake.wire
    quake.sink %ry : !quake.wire
    quake.sink %rz : !quake.wire
    return
  }
}
"""

CONTROLLED_RZ = r"""
module {
  func.func @__nvqpp__mlirgen__crz() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = arith.constant 0.5 : f64
    %0 = quake.null_wire
    %1 = quake.null_wire
    %r:2 = quake.rz (%a) [%0] %1 : (f64, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    quake.sink %r#0 : !quake.wire
    quake.sink %r#1 : !quake.wire
    return
  }
}
"""


def test_quake_import_rotation_logical_estimate():
    p0 = cudaq.logical.compiler.import_quake(ROTATIONS)
    assert cudaq.logical.analysis.estimate(
        p0, tier=cudaq.logical.analysis.Tier.LOGICAL).actions == {
            "qlx_standard_pauli_rotation": 3
        }


def test_quake_import_rejects_controlled_rotation():
    with pytest.raises(RuntimeError, match="controlled rotations"):
        cudaq.logical.compiler.import_quake(CONTROLLED_RZ)
