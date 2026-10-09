# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

include(CudaqPulseStagePython)

# Match CUDA-Q's build-tree Python layout. One PYTHONPATH entry exposes both
# the staged Python sources and the native extension.
file(GLOB_RECURSE _cudaq_pulse_python_sources
  RELATIVE "${CUDAQ_PULSE_FRONTEND_DIR}"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.py"
  "${CUDAQ_PULSE_FRONTEND_DIR}/*.pyi")
list(APPEND _cudaq_pulse_python_sources py.typed)

cudaq_pulse_stage_python_sources(CudaqPulsePythonStaging
  ROOT_DIR "${CUDAQ_PULSE_FRONTEND_DIR}"
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse"
  SOURCES ${_cudaq_pulse_python_sources})
add_dependencies(_cudaq_pulse_native CudaqPulsePythonStaging)

set(CUDAQ_PULSE_PYTHONPATH "${CMAKE_BINARY_DIR}/python")

set(_cudaq_pulse_build_targets
  cudaq-pulse-opt
  _cudaq_pulse_native
  CudaqPulsePythonStaging)
if(TARGET cudm_runtime)
  list(APPEND _cudaq_pulse_build_targets cudm_runtime)
endif()
add_custom_target(pulse DEPENDS ${_cudaq_pulse_build_targets})

install(DIRECTORY "${CUDAQ_PULSE_FRONTEND_DIR}"
  DESTINATION .
  COMPONENT CudaqPulse
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
