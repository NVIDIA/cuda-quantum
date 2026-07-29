# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CUDAQ_PULSE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CUDAQ_PULSE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CUDAQ_PULSE_FRONTEND_DIR
    "${CUDAQ_PULSE_SOURCE_DIR}/core/frontend/cudaq_pulse")
list(PREPEND CMAKE_MODULE_PATH "${CUDAQ_PULSE_SOURCE_DIR}/cmake")

include_directories(
  ${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/include
  ${CUDAQ_PULSE_BINARY_DIR}/core/mlir/include)

# Generate the three research-preview dialects.
add_subdirectory(core/mlir/include/cudaq-pulse)

# Build the dialects, transforms, conversions, C API, and optimizer driver.
add_subdirectory(core/mlir)
add_subdirectory(core/mlir/tools/cudaq-pulse-opt)

# Resolve the optional GPU runtime plus the Python and nanobind dependencies.
include(CudaqPulseDependencies)
add_subdirectory(core/mlir/bindings)

# Stage the Python package and register the aggregate build/install target.
include(CudaqPulsePython)

# Keep test and documentation target definitions out of the package entrypoint.
include(CudaqPulseTesting)
include(CudaqPulseDocumentation)
