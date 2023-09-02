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

set(LLVM_TARGET_DEFINITIONS options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(NvqppDriverOptions)

set(TOOL_NAME nvq++-v1)
configure_file("nvqpp_flag_configs.h.in" "nvqpp_flag_configs.h" @ONLY)
add_llvm_executable(${TOOL_NAME} nvqpp.cpp nvqpp_args.cpp nvqpp_driver.cpp nvqpp_options.cpp frontend_actions.cpp)
llvm_map_components_to_libnames(LLVM_LIBS support target option)
find_package(Clang REQUIRED CONFIG)
set(CLANG_RESOURCE_DIR  ${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR})
set(LLVM_BIN_DIR  ${LLVM_LIBRARY_DIR}/../bin)
target_compile_definitions(${TOOL_NAME} PRIVATE CLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR}")
target_compile_definitions(${TOOL_NAME} PRIVATE LLVM_BIN_DIR="${LLVM_BIN_DIR}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})


llvm_update_compile_flags(${TOOL_NAME})
target_link_libraries(${TOOL_NAME}
  PRIVATE
  ${LLVM_LIBS} 
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
  MLIRExecutionEngine
  MLIRTargetLLVMIRExport
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  clangBasic 
  clangDriver 
  clangCodeGen
  clangFrontendTool
  clangFrontend
  fmt::fmt-header-only
  CCDialect
  QuakeDialect
  OptCodeGen
  OptTransforms
  cudaq-mlirgen
)

mlir_check_all_link_libraries(${TOOL_NAME})
target_include_directories(${TOOL_NAME} PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})