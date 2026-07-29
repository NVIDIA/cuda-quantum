# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

nanobind_add_module(
  _cudaq_pulse_native
  NB_STATIC
  bindings.cpp
)

target_link_libraries(_cudaq_pulse_native PRIVATE
  CudaqPulsePulse
  CudaqPulsePulseTransforms
  CudaqPulseQOp
  CudaqPulseCuDensityMat
  CudaqPulsePulseToQOp
  CudaqPulseQOpToCuDensityMat
  CudaqPulseCuDensityMatToLLVM
  MLIRParser
  MLIRPass
  MLIRIR
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRReconcileUnrealizedCasts
  MLIRTransformUtils
)

set_target_properties(_cudaq_pulse_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_native"
)

install(
  TARGETS _cudaq_pulse_native
  LIBRARY DESTINATION cudaq_pulse/_native
)
