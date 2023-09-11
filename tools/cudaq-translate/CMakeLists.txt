# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

add_llvm_executable(cudaq-translate cudaq-translate.cpp)

llvm_update_compile_flags(cudaq-translate)
target_link_libraries(cudaq-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
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

  CCDialect
  OptCodeGen
  OptTransforms
  QuakeDialect
)

mlir_check_all_link_libraries(cudaq-translate)

install(TARGETS cudaq-translate DESTINATION bin)
