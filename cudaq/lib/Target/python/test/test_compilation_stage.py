# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

from cudaq.core._target import (CompilationStage, CustomPipelineLowering,
                                QuakeReferenceSemanticsProfile,
                                UncheckedProfile)



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



if __name__ == "__main__":
    pytest.main([__file__])
