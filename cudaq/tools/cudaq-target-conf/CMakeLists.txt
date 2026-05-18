# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})

set(EXE_NAME cudaq-target-conf)
add_llvm_executable(${EXE_NAME} cudaq-target-conf.cpp)
target_link_libraries(${EXE_NAME} PUBLIC CUDAQTargetConfigUtil)
install(TARGETS ${EXE_NAME} DESTINATION bin)
