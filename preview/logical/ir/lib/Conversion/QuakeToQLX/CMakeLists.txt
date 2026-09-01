# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Quake import goes through CUDA-Q's shared compiler target. This deliberately
# has no static source/build-tree fallback: CUDA-Q and QLX must resolve Quake,
# CC, LLVM, and MLIR through the same library and TypeIDs.
if(NOT CUDAQ_INCLUDE_DIR OR NOT IS_DIRECTORY "${CUDAQ_INCLUDE_DIR}")
  message(FATAL_ERROR
    "CUDAQ_INCLUDE_DIR must name the installed CUDA-Q headers")
endif()
add_library(QLXCUDAQuantumDialects INTERFACE)
target_link_libraries(QLXCUDAQuantumDialects INTERFACE cudaq::MLIR)
target_include_directories(QLXCUDAQuantumDialects INTERFACE
  "${CUDAQ_INCLUDE_DIR}")

add_mlir_conversion_library(MLIRQuakeToQLX
  ExpandQuakeRegisterTraversals.cpp
  PruneDeadCCLoopCarries.cpp
  QuakeToQLX.cpp
  QuakeToQLXPipelineReg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Conversion

  DEPENDS
  MLIRQLXIncGen
  MLIRQuakeToQLXPassesIncGen

  LINK_LIBS PUBLIC
  MLIRQLXDialect
  MLIRLVMDialect
  QLXCUDAQuantumDialects
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRArithDialect
  MLIRComplexDialect
  MLIRControlFlowInterfaces
  MLIRLoopLikeInterface
  MLIRTransforms
  MLIRUBDialect
)

qlx_propagate_llvm_rtti_eh(MLIRQuakeToQLX)


# Loadable DialectPlugin+PassPlugin so the Python path can convert Quake
# in-process (via _qlxRuntime.load_plugin + run_pass) instead of shelling out
# to the qlx-opt binary.
add_subdirectory(Plugin)
