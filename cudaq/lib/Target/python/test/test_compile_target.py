# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

from cudaq.core._target import (Codegen, CompilationStage, CompileTarget,
                                CustomPipelineLowering,
                                QuakeReferenceSemanticsProfile,
                                UncheckedProfile)


@pytest.mark.parametrize("target", [
    CompileTarget(),
    CompileTarget(ir_output_profile=UncheckedProfile()),
    CompileTarget(custom_pipeline="cse")
])
def test_defaults_on_every_constructor(target):
    assert target.fully_specialize
    assert target.argument_synth_change_semantics
    assert target.codegen == Codegen()


def test_default_stages_are_empty():
    assert CompileTarget().stages == []


def test_ir_output_profile_shortcut():
    profile = QuakeReferenceSemanticsProfile(basis_gates=["h"])
    expected = CompileTarget(stages=[CompilationStage(profile)],
                             codegen="qir-base")
    assert CompileTarget(ir_output_profile=profile,
                         codegen="qir-base") == expected


@pytest.mark.parametrize("pipeline, lowering", [
    ("cse", CustomPipelineLowering("cse")),
    (CustomPipelineLowering(high_level_pipeline="cse"),
     CustomPipelineLowering(high_level_pipeline="cse")),
])
def test_custom_pipeline_shortcut(pipeline, lowering):
    expected = CompileTarget(
        stages=[CompilationStage(UncheckedProfile(), lowering=lowering)],
        codegen="qir-base")
    assert CompileTarget(custom_pipeline=pipeline,
                         codegen="qir-base") == expected


@pytest.mark.parametrize("kwargs", [
    dict(ir_output_profile=QuakeReferenceSemanticsProfile(),
         custom_pipeline="cse"),
])
def test_shortcuts_are_exclusive(kwargs):
    with pytest.raises(TypeError):
        CompileTarget(**kwargs)


if __name__ == "__main__":
    pytest.main([__file__])
