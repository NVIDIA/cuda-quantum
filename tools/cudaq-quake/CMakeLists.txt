# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})

add_llvm_executable(cudaq-quake cudaq-quake.cpp)

llvm_update_compile_flags(cudaq-quake)
target_link_libraries(cudaq-quake
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIROptLib
  MLIRExecutionEngine
  MLIRTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation

  clangCodeGen
  clangFrontendTool
  clangFrontend
  fmt::fmt-header-only

  CCDialect
  QuakeDialect
  cudaq-mlirgen
)

mlir_check_all_link_libraries(cudaq-quake)

install(TARGETS cudaq-quake DESTINATION bin)
