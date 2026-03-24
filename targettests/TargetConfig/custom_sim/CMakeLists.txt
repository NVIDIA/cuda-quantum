# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This CMakeLists.txt just generates a default target config yml file
# for a custom simulator backend. We then use lit to validate the generated yml file.
include(${CMAKE_SOURCE_DIR}/cmake/Modules/NVQIRTargetConfigGen.cmake)
nvqir_write_target_config(MyCustomSimulator true "${CMAKE_BINARY_DIR}/targets/MyCustomSimulator.yml")
