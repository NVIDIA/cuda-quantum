# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

from cudaq._target import (Codegen, CompilationStage, CompileTarget,
                           CustomPipelineLowering,
                           QuakeReferenceSemanticsProfile, UncheckedProfile)


@pytest.mark.parametrize("target", [
    CompileTarget(),
    CompileTarget(ir_output_profile=UncheckedProfile()),
    CompileTarget(custom_pipeline="cse")
])
def test_defaults_on_every_constructor(target):
    assert not target.fully_specialize
    assert not target.argument_synth_change_semantics
    assert target.codegen == Codegen("qir:")


@pytest.mark.parametrize("flag", ["erase_noise", "erase_qec"])
def test_erasure_flags_change_identity(flag):
    target = CompileTarget()
    assert getattr(target, flag)
    changed = CompileTarget()
    setattr(changed, flag, False)
    assert changed != target
    assert hash(changed) != hash(target)


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
    (CustomPipelineLowering(high_level="cse"),
     CustomPipelineLowering(high_level="cse")),
])
def test_custom_pipeline_shortcut(pipeline, lowering):
    expected = CompileTarget(
        stages=[CompilationStage(UncheckedProfile(), lowering=lowering)],
        codegen="qir-base")
    assert CompileTarget(custom_pipeline=pipeline,
                         codegen="qir-base") == expected


@pytest.mark.parametrize("kwargs", [
    dict(stages=[], ir_output_profile=QuakeReferenceSemanticsProfile()),
    dict(stages=[], custom_pipeline="cse"),
    dict(ir_output_profile=QuakeReferenceSemanticsProfile(),
         custom_pipeline="cse"),
])
def test_shortcuts_are_exclusive(kwargs):
    with pytest.raises(TypeError):
        CompileTarget(**kwargs)


def test_string_converts_to_codegen():
    assert CompileTarget(codegen="qir-base").codegen == Codegen("qir-base")
    target = CompileTarget()
    target.codegen = "qir-base"
    assert target.codegen == Codegen("qir-base")


def test_codegen_edits_apply_in_place():
    target = CompileTarget(codegen="qir-base")
    target.codegen.post_codegen_passes = "canonicalize"
    assert target.codegen == Codegen("qir-base", "canonicalize")


if __name__ == "__main__":
    pytest.main([__file__])
