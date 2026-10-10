# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q target and backend-stack coverage for the P0-through-P3 preview."""

import pytest

import cudaq
import cudaq.logical
from cudaq._experimental import CustomTarget


@cudaq.logical.program
def bell_pair(a: cudaq.logical.types.logical_qubit,
              b: cudaq.logical.types.logical_qubit) -> tuple[bool, bool]:
    a = cudaq.logical.h(a)
    a, b = cudaq.logical.cx(a, b)
    return cudaq.logical.measure_z(a), cudaq.logical.measure_z(b)


def test_target_definition_advertises_only_real_recipes():
    target = cudaq.logical.targets.Target.define(
        "ir_only",
        emit_circuit=cudaq.logical.targets.LoweringSpec(
            (), cudaq.logical.targets.emit_mlir()),
    )

    assert target.capabilities() == ("emit_text", "emit_circuit")
    assert hasattr(target, "emit_text")
    assert hasattr(target, "emit_circuit")
    assert not hasattr(target, "sample")
    assert target.manifest()["recipes"]["emit_text"]["stages"] == ()


def test_logical_targets_importable():
    from cudaq.logical import targets

    assert hasattr(targets, "Target")
    assert hasattr(targets, "CliffordTBackend")


def test_target_define_exposes_a_one_level_backend_endpoint():
    target = cudaq.logical.targets.Target.define(
        "endpoint_ir",
        emit_text=cudaq.logical.targets.LoweringSpec(
            (), cudaq.logical.targets.emit_mlir()),
    )

    endpoint = target.runtime_endpoint

    assert isinstance(endpoint, cudaq.logical.targets.ProgramBackend)
    assert endpoint.spec.accepted_stages == ("CUDA-Q / Quake",)
    terminal = endpoint.next_backend
    assert isinstance(terminal, cudaq.logical.targets.TerminalBackend)
    assert terminal.specs["emit_text"] is target._specs["emit_text"]
    assert terminal.next_backend is None
    assert cudaq.logical.emit(bell_pair, target=target).startswith("module")


def test_target_print_stack_reports_its_runtime_backends(capsys):
    assert cudaq.logical.targets.mlir.print_stack() is None

    assert capsys.readouterr().out == (
        "== mlir :: CUDA-Q Logical backend stack =====================\n"
        "---- CUDA-Q / QUAKE -> P0 --------------------------------\n"
        "  ProgramBackend\n"
        "    source: CUDA-Q / Quake\n"
        "---- launch policies ---------------------------------------\n"
        "  EmitPolicy\n")

    cudaq.logical.targets.surface_target(logical_capacity=1).print_stack()
    assert capsys.readouterr().out == (
        "== surface :: CUDA-Q Logical backend stack =====================\n"
        "---- CUDA-Q / QUAKE -> P0 --------------------------------\n"
        "  ProgramBackend\n"
        "    source: CUDA-Q / Quake\n"
        "---- P0 -> P0 --------------------------------------------\n"
        "  CliffordTBackend\n"
        "    gate set: H/S/T/CX\n"
        "    precision: 0.0001\n"
        "---- P0 -> P1 --------------------------------------------\n"
        "  LogicalMachineBackend\n"
        "    machine: surfaceDeviceLogicalMachine\n"
        "    compute: capacity=1\n"
        "---- P1 -> P2 --------------------------------------------\n"
        "  QECMachineBackend\n"
        "    machine: surfaceDeviceQECMachine\n"
        "    compute: code=rotated_surface_3, d=3, blocks=1, "
        "architecture=Surface3Architecture\n"
        "---- launch policies ---------------------------------------\n"
        "  (none)\n")

    endpoint = cudaq.logical.targets.surface_target(
        logical_capacity=1).runtime_endpoint
    assert endpoint.spec.accepted_stages == ("CUDA-Q / Quake",)
    assert endpoint.next_backend.spec.accepted_stages == ("p0",)
    assert endpoint.next_backend.next_backend.spec.accepted_stages == ("p0",)
    assert endpoint.next_backend.next_backend.next_backend.spec.accepted_stages == (
        "p1",)


def test_target_eagerly_exposes_the_cudaq_target_configuration():
    target = cudaq.logical.targets.mlir
    compile_target = target.compile_target

    assert isinstance(target, CustomTarget)
    assert target.compile_target is compile_target
    assert compile_target.fully_specialize
    pipeline = compile_target.pipeline_config.override_pass_pipeline
    assert "prepare-for-wireset" in pipeline
    # Multi-control reduction needs reference semantics, so it must precede
    # the wire conversion inside `prepare-for-wireset`.
    assert pipeline.index("multicontrol-decomposition") < pipeline.index(
        "prepare-for-wireset")
    assert "prepare-quake-for-qlx" not in pipeline
    assert "convert-quake-to-qlx" not in pipeline


@cudaq.kernel
def logical_zero_readout():
    qubits = cudaq.qvector(1)
    mz(qubits[0])


@cudaq.kernel
def logical_zero_state():
    cudaq.qvector(1)


@cudaq.kernel
def logical_zero_memory(logical_qubits: int):
    qubits = cudaq.qvector(logical_qubits)
    mz(qubits)


@cudaq.kernel
def ccry_front_slice():
    qubits = cudaq.qvector(3)
    x(qubits[0])
    x(qubits[1])
    ry.ctrl(0.7, qubits.front(2), qubits[2])
    mz(qubits)


@cudaq.kernel
def cccry():
    qubits = cudaq.qvector(4)
    ry.ctrl(0.7, qubits[0], qubits[1], qubits[2], qubits[3])
    mz(qubits)


@cudaq.kernel
def ccrx():
    qubits = cudaq.qvector(3)
    rx.ctrl(0.7, qubits[0], qubits[1], qubits[2])
    mz(qubits)


@cudaq.kernel
def ccrz():
    qubits = cudaq.qvector(3)
    rz.ctrl(0.7, qubits[0], qubits[1], qubits[2])
    mz(qubits)


@cudaq.kernel
def ccr1():
    qubits = cudaq.qvector(3)
    r1.ctrl(0.7, qubits[0], qubits[1], qubits[2])
    mz(qubits)


@cudaq.kernel
def cch():
    qubits = cudaq.qvector(3)
    h.ctrl(qubits[0], qubits[1], qubits[2])
    mz(qubits)


@cudaq.kernel
def cccx():
    qubits = cudaq.qvector(4)
    x.ctrl(qubits[0], qubits[1], qubits[2], qubits[3])
    mz(qubits)


@cudaq.kernel
def ccry_body(qubits: cudaq.qview):
    ry.ctrl(0.7, qubits[0], qubits[1], qubits[2])


@cudaq.kernel
def ccry_adjoint():
    qubits = cudaq.qvector(3)
    cudaq.adjoint(ccry_body, qubits)
    mz(qubits)


@pytest.mark.parametrize(
    "kernel",
    [ccry_front_slice, cccry, ccrx, ccrz, ccr1, cch, cccx, ccry_adjoint],
    ids=lambda kernel: kernel.name)
def test_clifford_t_target_estimates_multi_controlled_gates(kernel):
    target = cudaq.logical.targets.clifford_t_target(precision=1.0e-4)

    try:
        cudaq.set_target(target)
        result = cudaq.estimate(kernel)
    finally:
        cudaq.reset_target()

    assert result.annotations


def test_estimator_target_estimates_a_doubly_controlled_ry():
    try:
        cudaq.set_target(cudaq.logical.targets.estimator)
        result = cudaq.estimate(ccry_front_slice)
    finally:
        cudaq.reset_target()

    assert result.annotations


def test_cudaq_selects_a_qlx_custom_target_for_estimation():
    target = cudaq.logical.targets.surface_target(logical_capacity=1)

    try:
        assert cudaq.set_target(target) is target
        estimates = cudaq.estimate(logical_zero_readout)
    finally:
        cudaq.reset_target()

    resources = cudaq.logical.estimate.FabricCounts.from_annotations(
        estimates.annotations)
    assert resources.patches_peak == 1


def test_cudaq_target_estimation_lowers_a_kernel_through_p3():
    target = cudaq.logical.targets.surface_physical_target(logical_capacity=1)
    assert isinstance(target.runtime_endpoint.next_backend,
                      cudaq.logical.targets.CliffordTBackend)
    assert target.runtime_endpoint.next_backend.precision == pytest.approx(
        1.0e-4)

    try:
        assert cudaq.set_target(target) is target
        result = cudaq.estimate(logical_zero_readout)
    finally:
        cudaq.reset_target()

    assert set(result.annotations) == {
        "LOGICAL",
        "STATIC",
        "ANALYTICAL",
        "SCHEDULE",
    }
    schedule = result.annotations["SCHEDULE"]
    assert schedule["physical_qubits"] == (
        cudaq.logical.codes.Surface[3].block.size)
    assert schedule["event_count"] > 0
    assert schedule["source_stage"] == "p3"


def test_target_estimate_options_can_pin_a_tier(monkeypatch):
    target = cudaq.logical.targets.surface_physical_target(logical_capacity=1)
    endpoint = target.runtime_endpoint
    monkeypatch.setattr(endpoint, "estimate_options", {
        **endpoint.estimate_options, "tier": "SCHEDULE"
    })
    builds = []
    estimate = endpoint.estimate
    monkeypatch.setattr(
        endpoint, "estimate", lambda build, *args, **kwargs:
        (builds.append(build), estimate(build, *args, **kwargs))[1])

    try:
        cudaq.set_target(target)
        result = cudaq.estimate(logical_zero_readout)
    finally:
        cudaq.reset_target()

    assert set(result.annotations) == {"SCHEDULE"}
    # An explicit tier still overrides the one the target pins.
    assert set(estimate(builds[0], tier="LOGICAL").annotations) == {"LOGICAL"}


def test_cudaq_target_estimation_traces_each_logical_phase():
    from cudaq.util import trace

    target = cudaq.logical.targets.surface_physical_target(logical_capacity=1)
    backend = trace.ChromeBackend()
    trace.set_backend(backend)
    try:
        cudaq.set_target(target)
        cudaq.estimate(logical_zero_readout)
    finally:
        trace.reset_backend()
        cudaq.reset_target()

    names = {e["name"] for e in backend.to_dict()["traceEvents"]}
    assert {
        f"cudaq.logical.target.{tag}" for tag in ("p0", "clifford_t", "p1",
                                                  "p2", "p3")
    } <= names
    assert {
        f"cudaq.estimate.{tier}" for tier in ("LOGICAL", "STATIC", "ANALYTICAL",
                                              "SCHEDULE")
    } <= names


def test_cudaq_p3_replays_destructive_multi_carrier_measurements():
    logical_qubits = 2
    target = cudaq.logical.targets.surface_physical_target(
        distance=3, logical_capacity=logical_qubits)

    try:
        cudaq.set_target(target)
        result = cudaq.estimate(logical_zero_memory, logical_qubits)
    finally:
        cudaq.reset_target()

    schedule = result.annotations["SCHEDULE"]
    assert schedule["physical_qubits"] == (
        logical_qubits * cudaq.logical.codes.Surface[3].block.size)
    assert schedule["event_counts"]["call_template"] > 0
    assert schedule["source_stage"] == "p3"


@pytest.mark.parametrize(("policy_name", "launch"), [
    ("sample", lambda: cudaq.sample(logical_zero_readout)),
    ("observe", lambda: cudaq.observe(logical_zero_state, cudaq.spin.z(0))),
    ("dem", lambda: cudaq.dem_from_kernel(logical_zero_readout)),
])
def test_cudaq_logical_reports_unsupported_preview_launch_policies(
        policy_name, launch):
    try:
        cudaq.set_target(cudaq.logical.targets.estimator)
        with pytest.raises(
                RuntimeError,
                match=rf"cudaq\.logical is in preview, and launch policy "
                rf"'{policy_name}' is not yet supported"):
            launch()
    finally:
        cudaq.reset_target()


def test_empty_terminal_reports_estimates_as_cudaq_annotations():
    endpoint = cudaq.logical.targets.TerminalBackend()
    result = endpoint.estimate(cudaq.logical.compile(qec_memory_program),
                               tier=cudaq.logical.estimate.Tier.LOGICAL)

    profile = cudaq.logical.estimate.LogicalProfile.from_annotations(
        result.annotations)
    assert profile.total_operations == 2


def test_backend_compiles_then_delegates_to_a_finalizer_leaf():
    calls = []

    class RecordingBackend(cudaq.logical.targets.Backend):

        def compile(self, build, **kwargs):
            calls.append(("compile", build.profile, kwargs["marker"]))
            return build

    def finish(module, _context, **kwargs):
        calls.append(("finalize", kwargs["marker"]))
        return "done"

    leaf = cudaq.logical.targets.Backend(
        cudaq.logical.targets.LoweringSpec((),
                                           finish,
                                           result_schema="text/plain"))
    root = RecordingBackend(cudaq.logical.targets.LoweringSpec((), None),
                            next_backend=leaf)

    assert root.emit(cudaq.logical.compile(bell_pair), marker="test") == "done"
    assert calls == [("compile", "p0", "test"), ("finalize", "test")]


def test_target_from_backend_preserves_the_explicit_runtime_endpoint():
    root = cudaq.logical.targets.ProgramBackend(
        next_backend=cudaq.logical.targets.TerminalBackend())

    target = cudaq.logical.targets.Target.from_backend("estimator", root)

    assert target.capabilities() == ()
    assert target.runtime_endpoint is root
    assert root.spec.produced_stage == "p0"


def test_predefined_qec_targets_currently_terminate_for_estimation_only():
    assert cudaq.logical.targets.estimator.capabilities() == ()
    assert isinstance(cudaq.logical.targets.estimator.runtime_endpoint,
                      cudaq.logical.targets.ProgramBackend)
    assert isinstance(
        cudaq.logical.targets.estimator.runtime_endpoint.next_backend,
        cudaq.logical.targets.TerminalBackend)

    clifford_t = cudaq.logical.targets.clifford_t
    assert clifford_t.capabilities() == ()
    assert isinstance(clifford_t.runtime_endpoint,
                      cudaq.logical.targets.ProgramBackend)
    assert isinstance(clifford_t.runtime_endpoint.next_backend,
                      cudaq.logical.targets.CliffordTBackend)

    for target in (
            cudaq.logical.targets.surface_target(),
            cudaq.logical.targets.steane_target(),
    ):
        assert target.capabilities() == ()
        assert target._device.qec is not None
        assert isinstance(target.runtime_endpoint,
                          cudaq.logical.targets.ProgramBackend)


def test_surface_target_links_gadgets_for_the_requested_distance():
    target = cudaq.logical.targets.surface_target(distance=5,
                                                  clifford_t_precision=1.0e-5)
    architecture = target._device.logical_to_qec[0].architecture

    assert architecture.code is cudaq.logical.codes.Surface[5]
    assert len(architecture.link_roots) == 7
    assert target.runtime_endpoint.next_backend.precision == 1.0e-5


def test_surface_physical_target_provisions_every_encoded_block():
    target = cudaq.logical.targets.surface_physical_target(
        distance=5,
        logical_capacity=3,
        clifford_t_precision=1.0e-5,
        cycle_time=2.0e-9,
    )
    device = target._device
    qec_region = device.logical_to_qec[0].qec_region
    carriers = device.qec_to_physical[0].resources[0]

    assert target.name == "surface_physical"
    assert qec_region.block_capacity == 3
    assert carriers.count == 3 * cudaq.logical.codes.Surface[5].block.size
    assert device.operating_point.timing["cycle_ns"] == pytest.approx(2.0)
    assert device.operating_point.calibration[
        "physical_error"] == pytest.approx(1.0e-3)
    assert target.runtime_endpoint.next_backend.precision == 1.0e-5


def test_none_clifford_t_precision_omits_the_synthesis_backend():
    target = cudaq.logical.targets.surface_target(clifford_t_precision=None)

    assert isinstance(target.runtime_endpoint.next_backend,
                      cudaq.logical.targets.LogicalMachineBackend)


def test_qec_target_factories_accept_an_optional_downstream_backend():
    terminal = cudaq.logical.targets.TerminalBackend()

    surface = cudaq.logical.targets.surface_target(next_backend=terminal)
    steane = cudaq.logical.targets.steane_target(next_backend=terminal)

    surface_tail = surface.runtime_endpoint.next_backend
    assert isinstance(surface_tail, cudaq.logical.targets.CliffordTBackend)
    assert surface_tail.next_backend.next_backend.next_backend is terminal
    assert (steane.runtime_endpoint.next_backend.next_backend.next_backend
            is terminal)


def test_steane_target_links_its_qec_realizations():
    target = cudaq.logical.targets.steane_target()
    architecture = target._device.logical_to_qec[0].architecture

    assert architecture.code is cudaq.logical.codes.Steane
    assert len(architecture.link_roots) == 7


@cudaq.logical.program
def qec_memory_program() -> bool:
    return cudaq.logical.measure_z(cudaq.logical.prepare_zero())


@cudaq.logical.program
def qec_standard_css_program() -> bool:
    qubit = cudaq.logical.prepare_plus()
    qubit = cudaq.logical.x(qubit)
    qubit = cudaq.logical.z(qubit)
    qubit = cudaq.logical.idle(qubit, rounds=1)
    return cudaq.logical.measure_x(qubit)


@pytest.mark.parametrize("factory, synthesis", [
    (cudaq.logical.targets.surface_target, True),
    (cudaq.logical.targets.steane_target, False),
])
def test_predefined_qec_targets_lower_through_their_linked_realisations(
        factory, synthesis):
    root = factory(logical_capacity=1).runtime_endpoint

    # The CSS recipe presently implements preparation, Pauli, memory, and
    # measurement objectives. Clifford+T gadget realizations are deliberately
    # outside this target-stack test; here we validate that synthesis occurs
    # before the existing P1/P2 lowering path.
    programs = (qec_memory_program,) if synthesis else (
        qec_memory_program,
        qec_standard_css_program,
    )
    for program in programs:
        p0 = root.compile(cudaq.logical.compile(program))
        backend = root.next_backend
        if synthesis:
            assert isinstance(backend, cudaq.logical.targets.CliffordTBackend)
            p0 = backend.compile(p0)
            assert p0.synthesis.gate_set == "clifford_t"
            backend = backend.next_backend
        p1 = backend.compile(p0)
        p2 = backend.next_backend.compile(p1)

        assert (p0.profile, p1.profile) == ("p0", "p1")
        assert p2.profile in {"p2a", "p2n"}


@cudaq.logical.program
def non_clifford_t_rotation() -> bool:
    qubit = cudaq.logical.rz(cudaq.logical.prepare_zero(), 0.3)
    return cudaq.logical.measure_z(qubit)


def test_clifford_t_backend_synthesizes_once_at_its_creation_precision():
    target = cudaq.logical.targets.clifford_t_target(precision=1.0e-4)
    backend = target.runtime_endpoint.next_backend

    synthesized = backend.compile(
        cudaq.logical.compile(non_clifford_t_rotation))

    assert synthesized.synthesis.gate_set == "clifford_t"
    assert synthesized.synthesis.precision == pytest.approx(1.0e-4)
    assert backend.compile(synthesized) is synthesized


def test_clifford_t_synthesis_is_available_from_the_compiler_facade():
    build = cudaq.logical.compiler.synthesize(
        non_clifford_t_rotation,
        gate_set=cudaq.logical.compiler.gate_sets.clifford_t,
        precision=1.0e-4,
    )

    assert build.synthesis.gate_set == "clifford_t"


def test_unversioned_custom_target_replay_fails_with_typed_unavailability():
    local = cudaq.logical.targets.Target.define(
        "local_only",
        emit_text=cudaq.logical.targets.LoweringSpec(
            (), cudaq.logical.targets.emit_mlir()),
    )

    with pytest.raises(cudaq.logical.targets.UnavailableTargetError,
                       match="no versioned"):
        cudaq.logical.targets.Target.replay(local.serialize())


def test_target_emission_preserves_frozen_snapshot_and_cached_inspection():
    build = cudaq.logical.compile(bell_pair)
    before = build.to_mlir()

    emitted = cudaq.logical.emit(build, target=cudaq.logical.targets.mlir)

    assert "qlx.program @bell_pair" in emitted
    assert build.to_mlir() == before
    assert build.module is build.module


def test_stage_free_target_lowering_reuses_build_verification_authority():
    from unittest.mock import patch

    from cudaq.logical.lower.driver import lower

    build = cudaq.logical.compile(bell_pair)
    with patch.object(
            type(build._module.operation),
            "verify",
            side_effect=AssertionError("stage-free clone was re-verified"),
    ):
        work, ctx = lower(
            cudaq.logical.targets.LoweringSpec(
                (), cudaq.logical.targets.emit_mlir()),
            build,
        )

    assert work is not build._module
    assert ctx._verification_receipt.module is work


def test_transforming_target_lowering_still_verifies_its_output():
    from unittest.mock import patch

    from cudaq.logical.lower.driver import lower
    from cudaq.logical.lower.stage import Stage

    class NoopStage(Stage):

        def apply(self, _module, _ctx):
            pass

    build = cudaq.logical.compile(bell_pair)
    with patch.object(
            type(build._module.operation),
            "verify",
            return_value=False,
    ) as verify:
        with pytest.raises(ValueError,
                           match="target lowering produced invalid"):
            lower(
                cudaq.logical.targets.LoweringSpec(
                    (NoopStage(),), cudaq.logical.targets.emit_mlir()),
                build,
            )

    verify.assert_called_once_with()
