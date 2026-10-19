# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudaq._target import (CompilationStage, CustomPipelineLowering,
                           QuakeReferenceSemanticsProfile, UncheckedProfile)


@pytest.mark.parametrize("fragment", ["high_level", "mid_level", "low_level"])
def test_lowering_pipeline_and_fragments_are_exclusive(fragment):
    with pytest.raises(ValueError):
        CustomPipelineLowering("cse", **{fragment: "canonicalize"})


@pytest.mark.parametrize(
    "profile",
    [UncheckedProfile(), QuakeReferenceSemanticsProfile()])
def test_stage_none_lowering_means_inferred(profile):
    stage = CompilationStage(profile)
    assert stage.lowering is None
    stage.lowering = CustomPipelineLowering("canonicalize")
    assert stage.lowering.override_pass_pipeline == "canonicalize"
    stage.lowering = None
    assert stage.lowering is None
    assert stage == CompilationStage(profile)


def test_stage_lowering_has_no_string_shortcut():
    with pytest.raises(TypeError):
        CompilationStage(UncheckedProfile(), lowering="canonicalize")


if __name__ == "__main__":
    pytest.main([__file__])
