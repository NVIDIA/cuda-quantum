# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(cudaq-pulse-opt
  cudaq-pulse-opt.cpp
)

target_link_libraries(cudaq-pulse-opt PRIVATE
  CudaqPulsePulse
  CudaqPulsePulseTransforms
  CudaqPulseQOp
  CudaqPulseCuDensityMat
  CudaqPulsePulseToQOp
  CudaqPulseQOpToCuDensityMat
  CudaqPulseCuDensityMatToLLVM
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLLVMDialect
  MLIROptLib
  MLIRPass
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransformUtils
)

llvm_update_compile_flags(cudaq-pulse-opt)
mlir_check_all_link_libraries(cudaq-pulse-opt)
