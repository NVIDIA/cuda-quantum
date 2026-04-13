# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Add the phase folding test targets.
configure_file(phase-folding-bench-mins.yml ${CMAKE_BINARY_DIR}/targets/phase-folding-bench-mins.yml COPYONLY)
configure_file(phase-folding-bench.yml ${CMAKE_BINARY_DIR}/targets/phase-folding-bench.yml COPYONLY)
