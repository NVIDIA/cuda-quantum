# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

from cudaq.core._target import (CompilationStage, CompileTarget,
                                CustomPipelineLowering, QuakeMappedProfile,
                                QuakeReferenceSemanticsProfile,
                                UncheckedProfile)


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


def test_quake_mapped_profile_repr():
    profile = QuakeMappedProfile(basis_gates=["h", "z(1)"], device="grid(2,2)")
    assert eval(repr(profile)) == profile
    assert profile != QuakeMappedProfile(basis_gates=["h", "z(1)"])
    assert repr(QuakeMappedProfile()).endswith("device=None)")


if __name__ == "__main__":
    pytest.main([__file__])
