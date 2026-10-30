# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

from cudaq._target import (Codegen, CompilationStage, CompileTarget, CostModel,
                           CustomPipelineLowering, QuakeMappedProfile,
                           QuakeReferenceSemanticsProfile, QuakeUnmappedProfile,
                           UncheckedProfile)


@pytest.mark.parametrize("make", [
    UncheckedProfile,
    lambda: QuakeReferenceSemanticsProfile(basis_gates=["z(1)"]),
    lambda: QuakeUnmappedProfile(basis_gates=["t"],
                                 approximation_tolerance=1e-3,
                                 cost_model=CostModel.t_count),
    lambda: QuakeMappedProfile(basis_gates=["z(1)"], device="path(4)"),
    lambda: CustomPipelineLowering("canonicalize"),
    lambda: Codegen("qir-base"),
    lambda: CompilationStage(QuakeReferenceSemanticsProfile()),
    lambda: CompileTarget(custom_pipeline="cse"),
])
def test_equal_values_hash_equal(make):
    left, right = make(), make()
    assert left == right
    assert hash(left) == hash(right)


def test_order_matters():
    ht = QuakeReferenceSemanticsProfile(basis_gates=["h", "t"])
    th = QuakeReferenceSemanticsProfile(basis_gates=["t", "h"])
    assert ht != th
    stages = [CompilationStage(ht), CompilationStage(th)]
    reversed_stages = list(reversed(stages))
    assert CompileTarget(stages=stages) != CompileTarget(stages=reversed_stages)


def test_inferred_differs_from_empty_custom_lowering():
    inferred = CompilationStage(UncheckedProfile())
    custom = CompilationStage(UncheckedProfile(),
                              lowering=CustomPipelineLowering())
    assert inferred != custom


def test_capability_flags_affect_equality():
    a, b = CompileTarget(), CompileTarget()
    b.replace_state_with_kernel = True
    assert a != b


def test_profile_repr():
    profile = QuakeReferenceSemanticsProfile(basis_gates=["h", "z(1)"],
                                             measurement_type="single_qubit",
                                             reset_after_measure=True)
    assert repr(profile) == (
        "QuakeReferenceSemanticsProfile(basis_gates=['h', 'z(1)'], "
        "measurement_type='single_qubit', reset_after_measure=True)")


def test_quake_unmapped_profile_repr():
    profile = QuakeUnmappedProfile(basis_gates=["h", "t"],
                                   approximation_tolerance=1e-10,
                                   cost_model=CostModel.t_count,
                                   unroll_loops="index_use",
                                   max_unroll_iterations=2048000)
    assert repr(profile) == (
        "QuakeUnmappedProfile(basis_gates=['h', 't'], "
        "approximation_tolerance=1e-10, cost_model=CostModel.t_count, "
        "unroll_loops='index_use', "
        "max_unroll_iterations=2048000)")
    assert eval(repr(profile)) == profile
    assert repr(QuakeUnmappedProfile()).endswith(
        "cost_model=None, unroll_loops='all', "
        "max_unroll_iterations=1024)")


@pytest.mark.parametrize("option, value", [
    ("unroll_loops", "index_use"),
    ("max_unroll_iterations", 2048000),
])
def test_quake_unmapped_option_changes_identity(option, value):
    baseline = QuakeUnmappedProfile()
    changed = QuakeUnmappedProfile(**{option: value})
    assert getattr(changed, option) == value
    assert changed != baseline
    assert hash(changed) != hash(baseline)
    setattr(changed, option, getattr(baseline, option))
    assert changed == baseline


def test_quake_unmapped_unroll_loops_rejects_unknown_value():
    with pytest.raises(ValueError):
        QuakeUnmappedProfile(unroll_loops="index-use")
    with pytest.raises(ValueError):
        QuakeUnmappedProfile().unroll_loops = "index-use"


def test_quake_unmapped_cost_model_accepts_name():
    profile = QuakeUnmappedProfile(cost_model="t_count")
    assert profile.cost_model == CostModel.t_count
    profile.cost_model = "two_qubit_gate_count"
    assert profile.cost_model == CostModel.two_qubit_gate_count
    profile.cost_model = None
    assert profile.cost_model is None
    with pytest.raises(ValueError):
        QuakeUnmappedProfile(cost_model="t-count")


def test_quake_mapped_profile_repr():
    profile = QuakeMappedProfile(basis_gates=["h", "z(1)"], device="grid(2,2)")
    assert eval(repr(profile)) == profile
    assert profile != QuakeMappedProfile(basis_gates=["h", "z(1)"])
    assert repr(QuakeMappedProfile()).endswith("device=None)")


def test_stage_repr():
    assert repr(CompilationStage(UncheckedProfile())) == (
        "CompilationStage(ir_output_profile=UncheckedProfile(), "
        "lowering=inferred)")
    custom = CompilationStage(UncheckedProfile(),
                              lowering=CustomPipelineLowering("cse"))
    assert repr(custom) == (
        "CompilationStage(ir_output_profile=UncheckedProfile(), "
        "lowering=custom-pipeline: CustomPipelineLowering(pipeline='cse', "
        "high_level='', mid_level='', low_level=''))")


if __name__ == "__main__":
    pytest.main([__file__])
