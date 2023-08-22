# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CUDAQ_RELOCATABLE_BOOL "false")
if(CUDAQ_BUILD_RELOCATABLE_PACKAGE) 
  set(CUDAQ_RELOCATABLE_BOOL "true")
endif()

set(CUDAQ_IS_APPLE "false")
if (APPLE) 
  set(CUDAQ_IS_APPLE "true")
endif()

set(TOOL_NAME nvq++-v1)
configure_file("nvqpp_flag_configs.h.in" "nvqpp_flag_configs.h" @ONLY)
add_llvm_executable(${TOOL_NAME} nvqpp.cpp)
llvm_map_components_to_libnames(LLVM_LIBS support target option)
find_package(Clang REQUIRED CONFIG)
set(CLANG_RESOURCE_DIR  ${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR})
set(LLVM_BIN_DIR  ${LLVM_LIBRARY_DIR}/../bin)
target_compile_definitions(${TOOL_NAME} PRIVATE CLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR}")
target_compile_definitions(${TOOL_NAME} PRIVATE LLVM_BIN_DIR="${LLVM_BIN_DIR}")



get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})


llvm_update_compile_flags(${TOOL_NAME})
target_link_libraries(${TOOL_NAME}
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

mlir_check_all_link_libraries(${TOOL_NAME})



target_link_libraries(${TOOL_NAME} PRIVATE
  ${LLVM_LIBS} 
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  clangBasic 
  clangDriver 
  clangFrontend

)

target_include_directories(${TOOL_NAME} PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})