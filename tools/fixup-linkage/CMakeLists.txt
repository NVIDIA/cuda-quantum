# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
)

add_llvm_executable(fixup-linkage fixup-linkage.cpp)

llvm_update_compile_flags(fixup-linkage)

install(TARGETS fixup-linkage DESTINATION bin)
