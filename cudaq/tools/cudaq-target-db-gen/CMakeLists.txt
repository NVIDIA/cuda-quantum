# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})

set(EXE_NAME cudaq-target-db-gen)
add_llvm_executable(${EXE_NAME} cudaq-target-db-gen.cpp)
target_link_libraries(${EXE_NAME} PUBLIC CUDAQTargetConfigYaml)
# Installed (unlike the in-tree-only use of this tool during CUDA-Q's own
# build) so that external plugin authors can compile their own targets
# through it - see docs/sphinx/using/extending/packaging.rst and
# cudaq-target-db-gen --plugin.
install(TARGETS ${EXE_NAME} DESTINATION bin COMPONENT Nvqpp)
