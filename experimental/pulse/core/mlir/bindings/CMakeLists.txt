# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

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
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIRTransformUtils
)

install(
  TARGETS _cudaq_pulse_native
  LIBRARY DESTINATION cudaq_pulse/_native
)
