# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import pytest

import cudaq.logical as qlx
import cudaq.logical.compiler.linearity as linearity
import cudaq.mlir.ir as mlir_ir
from cudaq.logical.compiler.linearity import (
    LinearityError,
    check_linearity,
    is_linear_type,
    verify_linearity,
)


@qlx.code
class Steane:
    block = qlx.codes.CSSBlock(data=7, sx=3, sz=3)
    d = 3
    hx = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6))
    hz = hx
    lx = (tuple(range(7)),)
    lz = lx


trivial_code = """
  fabric.code @c {distance = 1 : i64, n = 1 : i64, k = 1 : i64, hx = [],
                  hz = [], lx = [array<i64: 0>], lz = [array<i64: 0>],
                  partitions = {data = 1 : i64}}
"""


def parse(body: str) -> mlir_ir.Module:
    return mlir_ir.Module.parse("module {" + trivial_code + body + "}",
                                mlir_ir.Context())


def test_normal_program_and_gadget_bodies_verify_single_ownership():

    @qlx.program
    def byproduct() -> bool:
        q = qlx.prepare_zero()
        q, parity = qlx.mpp(qlx.types.Z(q))
        with qlx.ops.if_(parity, carries=(q,)) as branch:
            with branch.then():
                branch.yield_(qlx.x(q))
            with branch.else_():
                branch.yield_(q)
        q, = branch.results
        return qlx.measure_z(q)

    report = check_linearity(qlx.compile(byproduct).module)
    # The same qubit is legitimately consumed once in each exclusive branch;
    # the analysis must accept the canonical Pauli-byproduct pattern.
    assert report.result == "pass"
    assert report.checked_bodies == ("qlx.program @byproduct",)
    assert report.linear_values > 0

    @qlx.gadget(implements=qlx.logical.idle)
    def idle_round(block: qlx.patch[Steane]) -> qlx.patch[Steane]:
        block, _ = qlx.extract_syndrome(block, record="round")
        return block

    gadget_report = check_linearity(qlx.compile(idle_round).module)
    assert gadget_report.result == "pass"
    assert "fabric.gadget @idle_round" in gadget_report.checked_bodies


def test_passing_analysis_does_not_render_diagnostic_value_names(monkeypatch):
    module = parse("""
  fabric.gadget @g(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %0 = fabric.h %arg0 data : !fabric.patch<@c>
    fabric.return %0 : !fabric.patch<@c>
  }
""")

    def unexpected_description(*_args):
        raise AssertionError("passing values must not render diagnostic names")

    monkeypatch.setattr(linearity, "_describe_value", unexpected_description)
    assert check_linearity(module).result == "pass"


def test_double_consumed_patch_fails_verification_with_a_clear_message():
    # Plain MLIR SSA accepts this module (two users of %arg0), which is
    # exactly why the dedicated linear-use analysis must reject it.
    module = parse("""
  fabric.gadget @g(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %0 = fabric.h %arg0 data : !fabric.patch<@c>
    %1 = fabric.h %arg0 data : !fabric.patch<@c>
    fabric.return %0 : !fabric.patch<@c>
  }
""")
    assert module.operation.verify()  # ordinary SSA verification passes

    report = check_linearity(module)
    assert report.result == "fail"
    kinds = {violation.kind for violation in report.violations}
    assert "double-consume" in kinds
    # %1 is produced by the second (illegal) consume and then dropped.
    assert "unconsumed" in kinds

    with pytest.raises(LinearityError, match="consumed more than once"):
        verify_linearity(module, subject="gadget @g")
    with pytest.raises(LinearityError, match="%arg0"):
        verify_linearity(module)


def test_use_after_destructive_measure_fails_at_the_ir_level():
    # Complete data measurement is terminal at P2, so the dialect verifier
    # rejects this before the generic linear-use analysis needs to run.
    with pytest.raises(
            mlir_ir.MLIRError,
            match=
            "complete data measurement must have exactly one terminal disposal use",
    ):
        parse("""
  fabric.gadget @m(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %out, %bits = fabric.mz %arg0 data : !fabric.patch<@c> -> tensor<1xi1>
    %1 = fabric.h %arg0 data : !fabric.patch<@c>
    fabric.return %1 : !fabric.patch<@c>
  }
""")

    # The portable layer agrees: measuring a logical qubit consumes it.
    program = mlir_ir.Module.parse(
        """
module {
  qlx.program @p : () -> i1 attributes {qlx.profile = "p0", qlx.stage = "p0"} {
    %q = qlx.prepare "zero" : !qlx.logical_qubit
    %a = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
    %b = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
    %x = qlx.xor %a, %b : i1
    qlx.return %x : i1
  }
}
""",
        mlir_ir.Context(),
    )
    with pytest.raises(LinearityError, match="qlx.measure"):
        verify_linearity(program)


def test_linear_state_may_not_be_consumed_inside_an_unowned_loop_body():
    module = parse("""
  fabric.gadget @l(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    cflow.repeat 3 iter() {
      %h = fabric.h %arg0 data : !fabric.patch<@c>
      cflow.yield
    }
    fabric.return %arg0 : !fabric.patch<@c>
  }
""")
    report = check_linearity(module)
    assert report.result == "fail"
    assert any(violation.kind == "loop-capture-consume"
               for violation in report.violations)


def test_branch_exclusive_consumption_is_one_dynamic_owner():
    module = parse("""
  fabric.gadget @b(%arg0: !fabric.patch<@c>, %cond: i1)
      -> !fabric.patch<@c> {
    %r = cflow.if %cond -> !fabric.patch<@c> {
      %h = fabric.h %arg0 data : !fabric.patch<@c>
      cflow.yield %h : !fabric.patch<@c>
    } else {
      cflow.yield %arg0 : !fabric.patch<@c>
    }
    fabric.return %r : !fabric.patch<@c>
  }
""")
    assert check_linearity(module).result == "pass"

    # ... but consuming in a branch and again after the op is a double use.
    escaped = parse("""
  fabric.gadget @e(%arg0: !fabric.patch<@c>, %cond: i1)
      -> !fabric.patch<@c> {
    %r = cflow.if %cond -> !fabric.patch<@c> {
      %h = fabric.h %arg0 data : !fabric.patch<@c>
      cflow.yield %h : !fabric.patch<@c>
    } else {
      cflow.yield %arg0 : !fabric.patch<@c>
    }
    %again = fabric.h %arg0 data : !fabric.patch<@c>
    fabric.dealloc %again : !fabric.patch<@c>
    fabric.return %r : !fabric.patch<@c>
  }
""")
    report = check_linearity(escaped)
    assert any(v.kind == "double-consume" for v in report.violations)


def test_event_observers_borrow_without_consuming():
    module = parse("""
  fabric.gadget @o(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %ev = fabric.resource_request "t_state" from @stream
        : !event.handle<!fabric.resource<@t_state>, "linear">
    %ready = event.test %ev
        : !event.handle<!fabric.resource<@t_state>, "linear"> -> i1
    %res = event.await %ev
        : !event.handle<!fabric.resource<@t_state>, "linear">
        -> !fabric.resource<@t_state>
    fabric.discard_resource %res : !fabric.resource<@t_state>
    fabric.return %arg0 : !fabric.patch<@c>
  }
""")
    # event_test observes; event_await is the single consume: no violation.
    assert check_linearity(module).result == "pass"

    stale = parse("""
  fabric.gadget @s(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %ev = fabric.resource_request "t_state" from @stream
        : !event.handle<!fabric.resource<@t_state>, "linear">
    %res = event.await %ev
        : !event.handle<!fabric.resource<@t_state>, "linear">
        -> !fabric.resource<@t_state>
    %ready = event.test %ev
        : !event.handle<!fabric.resource<@t_state>, "linear"> -> i1
    fabric.discard_resource %res : !fabric.resource<@t_state>
    fabric.return %arg0 : !fabric.patch<@c>
  }
""")
    report = check_linearity(stale)
    assert any(v.kind == "borrow-after-consume" for v in report.violations)


def test_leaked_linear_owner_is_reported():
    module = parse("""
  fabric.gadget @leak(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %0 = fabric.alloc {code = @c, region = @r} : !fabric.patch<@c>
    fabric.return %arg0 : !fabric.patch<@c>
  }
""")
    report = check_linearity(module)
    assert [violation.kind for violation in report.violations] == ["unconsumed"]


def test_linear_type_recognition_covers_the_contracted_families():
    context = mlir_ir.Context()

    def t(text):
        return mlir_ir.Type.parse(text, context)

    assert is_linear_type(t("!qlx.logical_qubit"))
    assert is_linear_type(t('!qlx.logical_resource<"t_state">'))
    assert is_linear_type(
        t('!event.handle<!qlx.logical_resource<"t_state">, "linear">'))
    assert is_linear_type(t('!event.handle<!fabric.resource<@t>, "linear">'))
    assert is_linear_type(t("!phys.state<@q0>"))
    assert is_linear_type(t("!phys.resource_payload<@t_state>"))
    assert not is_linear_type(t("i1"))
    assert not is_linear_type(t("!fabric.syndrome<@c>"))
    assert not is_linear_type(
        t('!event.handle<!fabric.resource<@t>, "shared">'))


def test_compiled_builds_carry_real_linearity_evidence():

    @qlx.program
    def one() -> bool:
        return qlx.measure_z(qlx.prepare_zero())

    build = qlx.compile(one)
    records = [
        record for record in build.evidence
        if record.kind == "linearity_verification"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.result == "pass"
    assert record.obligations == ("linear-ownership",)
    # The record is minted from an actual analysis run: it names the checker
    # and reports what it walked.
    assert "checker:qlx-linear-use/v1" in record.assumptions
    checked = [
        item for item in record.assumptions
        if item.startswith("checked-bodies:")
    ]
    assert checked and int(checked[0].split(":", 1)[1]) >= 1
    counted = [
        item for item in record.assumptions if item.startswith("linear-values:")
    ]
    assert counted and int(counted[0].split(":", 1)[1]) >= 1

    @qlx.gadget(implements=qlx.logical.idle)
    def evidence_idle(block: qlx.patch[Steane]) -> qlx.patch[Steane]:
        block, _ = qlx.extract_syndrome(block, record="round")
        return block

    gadget_build = qlx.compile(evidence_idle)
    gadget_records = [
        record for record in gadget_build.evidence
        if record.kind == "linearity_verification"
    ]
    assert len(gadget_records) == 1
    assert gadget_records[0].result == "pass"
    assert gadget_records[0].obligations == ("linear-patch-ownership",)
    # The unconditional gadget record no longer asserts linearity for free.
    static = [
        record for record in gadget_build.evidence
        if record.kind == "gadget_verification"
    ]
    assert static and "linear-patch-ownership" not in static[0].obligations


@qlx.machine
class ProjectionMachine:
    compute = qlx.architecture.Space(
        capabilities=(qlx.architecture.capability.logical_compute,), capacity=1)


projection_qubits = qlx.architecture.ResourceClass(
    "qubit",
    4,
    native_actions=("rpp",),
    native_instruments=(qlx.architecture.physical_instruments.MPP,),
)
projection_architecture = qlx.architecture.PhysicalMachine(
    "linearity_projection_architecture",
    resource_classes={"qubits": projection_qubits},
)

trivial_pair = qlx.codes.CSSCode(
    name="linearity_pair",
    n=2,
    k=2,
    d=1,
    block=qlx.codes.CSSBlock(data=2),
    lx=((0,), (1,)),
    lz=((0,), (1,)),
)
projection_device_builder = qlx.devices.DeviceBuilder(
    "ProjectionDevice",
    logical=ProjectionMachine,
    physical=projection_architecture,
)
projection_compute = projection_device_builder.qec.bind(
    projection_device_builder.logical.compute,
    encoding=trivial_pair,
)
projection_device_builder.physical.bind(
    projection_compute,
    to=projection_device_builder.physical.qubits,
)
ProjectionDevice = projection_device_builder.build()


def test_physical_projection_carries_real_linearity_evidence():
    # The P2->P3 projection route must discharge physical-state ownership
    # through the real linear-use analysis, exactly like compile(): the
    # unconditional projection record no longer asserts it for free.
    @qlx.objective
    def joint_parity_objective(
        left: qlx.types.logical_qubit,
        right: qlx.types.logical_qubit,
    ) -> tuple[qlx.types.logical_qubit, qlx.types.logical_qubit, bool]:
        return qlx.mpp(qlx.types.Z(left) @ qlx.types.Z(right))

    @qlx.gadget(implements=joint_parity_objective)
    def joint_parity(
        block: qlx.patch[trivial_pair],
    ) -> tuple[qlx.patch[trivial_pair], bool]:
        block, outcome = qlx.mpp(qlx.types.Z(block[0]) @ qlx.types.Z(block[1]))
        return block, outcome

    build = qlx.compile(
        joint_parity,
        pipeline=qlx.compiler.pipelines.physical(),
        device=ProjectionDevice,
    )
    assert build.profile == "p3"
    projection = [
        record for record in build.evidence
        if record.kind == "physical_projection_verification"
    ]
    assert projection and all(
        "linear-physical-state-ownership" not in record.obligations
        for record in projection)
    records = [
        record for record in build.evidence
        if record.kind == "linearity_verification" and record.obligations == (
            "linear-physical-state-ownership",)
    ]
    assert len(records) == 1
    record = records[0]
    assert record.result == "pass"
    assert "checker:phys.graph-physical-state-linearity/v1" in record.assumptions
    counted = [
        item for item in record.assumptions if item.startswith("linear-values:")
    ]
    assert counted and int(counted[0].split(":", 1)[1]) >= 1
