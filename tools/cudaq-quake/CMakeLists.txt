# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})

add_llvm_executable(cudaq-quake cudaq-quake.cpp)

llvm_update_compile_flags(cudaq-quake)
target_link_libraries(cudaq-quake
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
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
