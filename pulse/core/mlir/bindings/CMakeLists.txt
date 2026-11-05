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

# Unlike cudaq-pulse-opt, this extension has to share a process with `cudaq`,
# whose `libcudaqMLIR.so` already contains MLIR and LLVM. Linking the static
# MLIR archives here as well would give the process two copies of every LLVM
# command-line option and dialect registry, which aborts at import time with
# "Option ... registered more than once".
#
# So instead of linking the CudaqPulse* archives (whose interface pulls in the
# whole static MLIR closure), consume their object files directly and resolve
# every MLIR/LLVM symbol from libcudaqMLIR. Only MLIR libraries that
# libcudaqMLIR does not bundle are still linked statically.
target_link_libraries(_cudaq_pulse_native PRIVATE
  $<TARGET_OBJECTS:obj.CudaqPulsePulse>
  $<TARGET_OBJECTS:obj.CudaqPulsePulseTransforms>
  $<TARGET_OBJECTS:obj.CudaqPulseQOp>
  $<TARGET_OBJECTS:obj.CudaqPulseCuDensityMat>
  $<TARGET_OBJECTS:obj.CudaqPulsePulseToQOp>
  $<TARGET_OBJECTS:obj.CudaqPulseQOpToCuDensityMat>
  $<TARGET_OBJECTS:obj.CudaqPulseCuDensityMatToLLVM>
  # mlir::createReconcileUnrealizedCastsPass is the one symbol libcudaqMLIR
  # does not export. Link the archive by path rather than by target name: the
  # MLIRReconcileUnrealizedCasts target's link interface would pull the whole
  # static MLIR closure back in.
  $<TARGET_FILE:MLIRReconcileUnrealizedCasts>
  cudaq::MLIR
)

# $<TARGET_OBJECTS:> contributes object files but no build dependency.
add_dependencies(_cudaq_pulse_native
  obj.CudaqPulsePulse
  obj.CudaqPulsePulseTransforms
  obj.CudaqPulseQOp
  obj.CudaqPulseCuDensityMat
  obj.CudaqPulsePulseToQOp
  obj.CudaqPulseQOpToCuDensityMat
  obj.CudaqPulseCuDensityMatToLLVM)

# Resolve libcudaqMLIR at load time from the CUDA-Q wheel's library directory.
set_property(TARGET _cudaq_pulse_native APPEND PROPERTY
  BUILD_RPATH "${CUDAQ_LIBRARY_DIR}")
set_property(TARGET _cudaq_pulse_native APPEND PROPERTY
  INSTALL_RPATH "${CUDAQ_LIBRARY_DIR}")

# Fail the build if this extension references MLIR/LLVM symbols libcudaqMLIR
# does not export.
cudaq_check_mlir_symbol_closure(_cudaq_pulse_native)

set_target_properties(_cudaq_pulse_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_native"
)

install(
  TARGETS _cudaq_pulse_native
  LIBRARY DESTINATION cudaq_pulse/_native
  COMPONENT CudaqPulse
)
