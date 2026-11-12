# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for the experimental `CompileTarget` bindings.
#
# A compile target owns the compilation half of a backend: the pass pipelines,
# the code generation and the capabilities kernels are compiled against.
# Install it together with a runtime endpoint as a `CustomTarget`.

import os
import subprocess
import sys

import pytest

import cudaq
from cudaq._experimental import (Codegen, CompilationStage, CompileTarget,
                                 CostModel, CustomPipelineLowering,
                                 CustomTarget, LoopUnrolling, NoiseSupport,
                                 QuakeMappedProfile,
                                 QuakeReferenceSemanticsProfile,
                                 QuakeUnmappedProfile, UncheckedProfile)
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.ir import WalkResult

# Decompose swaps into CNOTs instead of running the default pipeline.
SWAP_TO_CX_PIPELINE = (
    "canonicalize,decomposition{enable-patterns=SwapToCX},canonicalize")


@pytest.fixture(autouse=True)
def reset_target():
    """Ensure every test starts and ends from a well-defined target."""
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def make_swap_kernel():
    # Build a fresh kernel per call so each has its own compiled-module cache
    # slot and cannot reuse a previously compiled artifact.

    @cudaq.kernel
    def swap_kernel():
        q = cudaq.qvector(2)
        x(q[0])
        swap(q[0], q[1])
        mz(q)

    return swap_kernel


def test_frontend_reexports_shared_bindings():
    from importlib import import_module
    from types import ModuleType
    from cudaq.core import _experimental as core_experimental
    from cudaq.core import results
    from cudaq.mlir._mlir_libs import _backends, _quakeDialects, _quakeDialectsCore

    assert Codegen is core_experimental.Codegen
    assert CompileTarget is core_experimental.CompileTarget
    assert CustomPipelineLowering is core_experimental.CustomPipelineLowering
    assert CustomTarget is core_experimental.CustomTarget
    assert cudaq.Resources is results.Resources
    assert cudaq.EstimateResult is results.EstimateResult
    # Every public native export must retain object identity through the
    # frontend, including imports using its original submodule paths.
    for source, destination in ((_quakeDialectsCore, _quakeDialects),
                                (_backends, cudaq_runtime)):
        for name, value in vars(source).items():
            if name.startswith("_"):
                continue
            assert getattr(destination, name) is value
            if isinstance(value, ModuleType):
                assert import_module(f"{destination.__name__}.{name}") is value


def test_get_compile_target_imports_only_core_types():
    code = """
import sys

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime

target = cudaq_runtime.get_compile_target()
assert type(target).__module__ == "cudaq.core._target"
assert "cudaq._experimental" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr


def round_trip(target):
    cudaq.set_target("qpp-cpu", compile_target=target)
    return cudaq_runtime.get_compile_target()


@pytest.mark.parametrize("noise_support", list(NoiseSupport))
def test_round_trip(noise_support):
    assert round_trip(CompileTarget()) == CompileTarget()

    target = CompileTarget(stages=[
        CompilationStage(
            UncheckedProfile(),
            CustomPipelineLowering(high_level_pipeline="hi",
                                   mid_level_pipeline="mid",
                                   low_level_pipeline="lo")),
        CompilationStage(UncheckedProfile(), CustomPipelineLowering("cse")),
        CompilationStage(
            QuakeReferenceSemanticsProfile(["h", "z(1)"], "single_qubit",
                                           True)),
        CompilationStage(QuakeReferenceSemanticsProfile()),
        CompilationStage(
            QuakeUnmappedProfile(["t"], 1e-3, CostModel.two_qubit_gate_count,
                                 LoopUnrolling.index_use, 7)),
        CompilationStage(QuakeUnmappedProfile()),
        CompilationStage(QuakeMappedProfile(["x"], "grid(2,2)")),
        CompilationStage(QuakeMappedProfile()),
    ],
                           codegen=Codegen("qir-base", "canonicalize"),
                           override_aot_compilation=True,
                           support_conditionals_on_measure_results=False,
                           support_device_calls=True,
                           support_resource_counts=False,
                           support_explicit_measurements=False,
                           support_observable_measurements=False,
                           support_sample_without_measurements=False,
                           fully_specialize=True,
                           preserve_loops=True,
                           noise_support=noise_support,
                           erase_qec=False,
                           argument_synth_change_semantics=True,
                           disable_qubit_mapping=True,
                           replace_state_with_kernel=True)
    assert round_trip(target) == target


def test_round_trip_converts_strings():
    target = round_trip(
        CompileTarget(codegen="qir-base",
                      noise_support="unsupported",
                      stages=[
                          CompilationStage(
                              QuakeUnmappedProfile(cost_model="t_count",
                                                   unroll_loops="index_use"))
                      ]))
    assert target == CompileTarget(
        codegen=Codegen("qir-base"),
        noise_support=NoiseSupport.unsupported,
        stages=[
            CompilationStage(
                QuakeUnmappedProfile(cost_model=CostModel.t_count,
                                     unroll_loops=LoopUnrolling.index_use))
        ])


def stage(profile, lowering=None):
    return CompileTarget(stages=[CompilationStage(profile, lowering)])


def fragments_with_override():
    """A lowering made invalid after construction, which the caster rejects."""
    lowering = CustomPipelineLowering("cse")
    lowering.high_level_pipeline = "cse"
    return stage(UncheckedProfile(), lowering)


def unknown_noise_support():
    """An enum name made invalid after construction, which the caster rejects."""
    target = CompileTarget()
    target.noise_support = "loud"
    return target


@pytest.mark.parametrize("target", [
    CompileTarget(stages=[42]),
    stage(QuakeReferenceSemanticsProfile(basis_gates="h")),
    fragments_with_override(),
    unknown_noise_support(),
    stage(UncheckedProfile(), "canonicalize"),
    stage(42),
    CompileTarget(fully_specialize=1),
])
def test_invalid_compile_target(target):
    with pytest.raises(TypeError):
        cudaq.set_target("qpp-cpu", compile_target=target)


def test_assigned_codegen_string_is_the_emission_format():
    target = CompileTarget()
    target.codegen = "qir-base"
    assert round_trip(target).codegen == Codegen("qir-base")


def test_custom_target_runtime_annotations():
    from dataclasses import fields
    from typing import get_type_hints
    from cudaq._experimental import RuntimeEndpoint
    from cudaq.core import _experimental as core_experimental

    expected = {
        "runtime_endpoint": RuntimeEndpoint,
        "compile_target": CompileTarget,
    }
    assert get_type_hints(CustomTarget) == expected
    assert {
        field.name: field.type for field in fields(CustomTarget)
    } == expected
    assert RuntimeEndpoint is core_experimental.RuntimeEndpoint


def swap_pipeline_target():
    return CompileTarget(custom_pipeline=SWAP_TO_CX_PIPELINE)


class CapturingEndpoint:
    """A runtime endpoint that keeps the MLIR it was handed."""

    def __init__(self):
        self.mlir_module = None

    def sample(self, module, args, **kwargs):
        self.mlir_module = module.mlir_module
        return cudaq.SampleResult()


def set_custom_target(compile_target, endpoint=None):
    if endpoint is None:
        endpoint = CapturingEndpoint()
    cudaq.set_target(
        CustomTarget(runtime_endpoint=endpoint, compile_target=compile_target))


def compiled_quake_ops(kernel, compile_target=None):
    """Launch `kernel` through a capturing endpoint and return its quake ops."""
    if compile_target is None:
        compile_target = CompileTarget()
    endpoint = CapturingEndpoint()
    set_custom_target(compile_target, endpoint=endpoint)
    cudaq.sample(kernel, shots_count=1)

    module = endpoint.mlir_module
    assert module is not None, "the endpoint was not handed a compiled module"

    ops = []

    def visit(op):
        if op.name.startswith("quake."):
            ops.append((op.name, len(op.operands)))
        return WalkResult.ADVANCE

    module.operation.walk(visit)
    return ops


def test_lowering_controls_compiled_ir():
    # Baseline: the default qpp-cpu pipeline keeps the swap intact.
    default_ops = compiled_quake_ops(make_swap_kernel())
    assert any(name == "quake.swap" for name, _ in default_ops)

    # Custom pipeline: decompose swap into CNOTs.
    decomposed_ops = compiled_quake_ops(make_swap_kernel(),
                                        compile_target=swap_pipeline_target())

    # The swap has been decomposed away into three controlled-x (CNOT) ops.
    assert all(name != "quake.swap" for name, _ in decomposed_ops)
    controlled_x = [
        name for name, num_operands in decomposed_ops
        if name == "quake.x" and num_operands > 1
    ]
    assert len(controlled_x) == 3


def test_compile_target_does_not_leak_after_switch():
    """A compile target must not survive a target change.

    Changing the target replaces the platform's QPUs, which drops the compile
    target with them. Otherwise a custom pipeline would leak into unrelated
    kernels (e.g. `cudaq.draw`).
    """

    def op_names(kernel, compile_target=None):
        return [name for name, _ in compiled_quake_ops(kernel, compile_target)]

    # Install a compile target that decomposes swaps into CNOTs.
    assert "quake.swap" not in op_names(make_swap_kernel(),
                                        compile_target=swap_pipeline_target())

    # Switching targets must restore default behaviour.
    cudaq.set_target("qpp-cpu")
    assert "quake.swap" in op_names(make_swap_kernel())

    # Same expectation after a reset.
    set_custom_target(swap_pipeline_target())
    cudaq.reset_target()
    assert "quake.swap" in op_names(make_swap_kernel())


def test_support_explicit_measurements():
    ct = CompileTarget()
    ct.support_explicit_measurements = False
    set_custom_target(ct)
    assert cudaq_runtime.supportsExplicitMeasurements() is False

    ct.support_explicit_measurements = True
    set_custom_target(ct)
    assert cudaq_runtime.supportsExplicitMeasurements() is True


# ---------------------------------------------------------------------------- #
# Use with a named target
# ---------------------------------------------------------------------------- #


def test_named_target_with_compile_target():
    ct = CompileTarget()
    ct.support_explicit_measurements = False
    cudaq.set_target("qpp-cpu", compile_target=ct)
    assert cudaq.get_target().name == "qpp-cpu"
    assert cudaq_runtime.supportsExplicitMeasurements() is False

    # A new seed must not rebuild the compile target from the target config.
    cudaq.set_random_seed(13)
    assert cudaq_runtime.supportsExplicitMeasurements() is False

    # Without compile_target, the target's default compile target is used.
    cudaq.set_target("qpp-cpu")
    assert cudaq_runtime.supportsExplicitMeasurements() is True


def test_named_target_compiles_against_given_compile_target():
    cudaq.set_target(
        "qpp-cpu",
        compile_target=CompileTarget(custom_pipeline="not-a-registered-pass"))
    with pytest.raises(RuntimeError, match="not-a-registered-pass"):
        cudaq.sample(make_swap_kernel(), shots_count=1)

    cudaq.set_target(cudaq.get_target("qpp-cpu"),
                     compile_target=swap_pipeline_target())
    assert cudaq.sample(make_swap_kernel(), shots_count=10)["01"] == 10


def test_staged_lowering():
    ct = CompileTarget(custom_pipeline=CustomPipelineLowering(
        high_level_pipeline=SWAP_TO_CX_PIPELINE))
    assert ct.stages[0].lowering.high_level_pipeline == SWAP_TO_CX_PIPELINE
    assert ct.stages[0].lowering.override_pass_pipeline == ""

    # The high-level pipeline runs around the built-in passes: the swap is
    # still decomposed into three controlled-x (CNOT) ops.
    decomposed_ops = compiled_quake_ops(make_swap_kernel(), compile_target=ct)
    assert all(name != "quake.swap" for name, _ in decomposed_ops)
    controlled_x = [
        name for name, num_operands in decomposed_ops
        if name == "quake.x" and num_operands > 1
    ]
    assert len(controlled_x) == 3


def make_static_bell_kernel():

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        mz(q)

    return bell


def compile_profile_target(kernel, controlled_gate):
    profile = QuakeReferenceSemanticsProfile(
        basis_gates=["h", "t", controlled_gate],
        measurement_type="single_qubit")
    target = CompileTarget(ir_output_profile=profile, codegen="qir-base")
    endpoint = CapturingEndpoint()
    set_custom_target(target, endpoint)
    cudaq.sample(kernel, shots_count=1)
    assert endpoint.mlir_module is not None
    ir = str(endpoint.mlir_module)
    assert ir.count("= quake.mz ") == 2
    assert not any(
        "quake.mz" in line and "!quake.veq" in line for line in ir.splitlines())
    assert "quake." + controlled_gate[0] + " [" in ir
    return ir


def test_profile_target_switching_uses_the_same_kernel_cache_safely():
    kernel = make_static_bell_kernel()
    z_ir = compile_profile_target(kernel, "z(1)")
    x_ir = compile_profile_target(kernel, "x(1)")
    assert "quake.x [" not in z_ir
    assert "quake.z [" not in x_ir
    assert compile_profile_target(kernel, "z(1)") == z_ir


def test_later_stage_cannot_add_passes_before_deploy():
    # Single-qubit measurements add `expand-measurements` before deploy, and
    # only the first stage may add passes there.
    stages = [
        CompilationStage(
            QuakeReferenceSemanticsProfile(basis_gates=["h", "t", gate],
                                           measurement_type="single_qubit"))
        for gate in ("z(1)", "x(1)")
    ]
    set_custom_target(CompileTarget(stages=stages, codegen="qir-base"))
    with pytest.raises(RuntimeError,
                       match="stage 2: only the first stage may add passes"):
        cudaq.sample(make_static_bell_kernel(), shots_count=1)


def remote_profile_target(post_codegen_passes=""):
    target = CompileTarget(ir_output_profile=QuakeReferenceSemanticsProfile(
        basis_gates=["h", "x(1)"], measurement_type="single_qubit"),
                           codegen=Codegen(
                               "qir-adaptive:0.1:int_computations",
                               post_codegen_passes=post_codegen_passes))
    target.fully_specialize = True
    return target


def test_remote_target_codegen_uses_given_compile_target():
    cudaq.set_target("quantinuum",
                     emulate=True,
                     compile_target=remote_profile_target())
    assert len(cudaq.sample(make_static_bell_kernel(), shots_count=10)) > 0

    cudaq.set_target(
        "quantinuum",
        emulate=True,
        compile_target=remote_profile_target("not-a-registered-pass"))
    with pytest.raises(RuntimeError, match="Could not successfully translate"):
        cudaq.sample(make_static_bell_kernel(), shots_count=10)


def test_remote_tracing_does_not_emit_target_code():
    cudaq.set_target(
        "quantinuum",
        emulate=True,
        compile_target=remote_profile_target("not-a-registered-pass"))
    kernel = make_static_bell_kernel()

    # Drawing executes the JIT artifact and must not run transport codegen.
    assert "h" in cudaq.draw(kernel)
    # Sampling the same kernel still requires transport codegen.
    with pytest.raises(RuntimeError, match="Could not successfully translate"):
        cudaq.sample(kernel, shots_count=1)


if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
