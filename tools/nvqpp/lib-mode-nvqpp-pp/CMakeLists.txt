# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_llvm_executable(nvq++-pp LibModeNvqppSourcePP.cpp 
  plugins/AddRegisterNamesVisitor.cpp
  plugins/EnforceObserveSpecification.cpp)

llvm_update_compile_flags(nvq++-pp)
target_link_libraries(nvq++-pp
  PRIVATE
  clangFrontendTool
  clangFrontend
  clangRewrite
  clangTooling
  cudaq-common
  fmt::fmt-header-only
)

mlir_check_all_link_libraries(nvq++-pp)

install(TARGETS nvq++-pp DESTINATION bin)