# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudaq._target import (Codegen, CompilationStage, CompileTarget,
                           CustomPipelineLowering,
                           QuakeReferenceSemanticsProfile, UncheckedProfile)


@pytest.mark.parametrize("make", [
    UncheckedProfile,
    lambda: QuakeReferenceSemanticsProfile(basis_gates=["z(1)"]),
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
