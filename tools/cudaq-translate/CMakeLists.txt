# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

add_llvm_executable(cudaq-translate cudaq-translate.cpp)

llvm_update_compile_flags(cudaq-translate)

cudaq_get_mlir_libs(MLIR_LIBS
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIROptLib
  MLIRTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  MLIRExecutionEngine
)

target_link_libraries(cudaq-translate
  PRIVATE
  ${MLIR_LIBS}
  # CUDAQ libraries
  CCDialect
  OptCodeGen
  OptTransforms
  QuakeDialect
)

mlir_check_all_link_libraries(cudaq-translate)

install(TARGETS cudaq-translate DESTINATION bin)
