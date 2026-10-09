# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

option(CUDAQ_PULSE_BUILD_DOCS
  "Build the CUDA-Q pulse research-preview docs" OFF)
if(NOT CUDAQ_PULSE_BUILD_DOCS)
  return()
endif()

get_filename_component(_cudaq_pulse_python_bin
  "${Python_EXECUTABLE}" DIRECTORY)
find_program(CUDAQ_PULSE_SPHINX_EXECUTABLE NAMES sphinx-build
  HINTS "${_cudaq_pulse_python_bin}"
  REQUIRED)
add_custom_target(pulse-docs
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${CUDAQ_PULSE_PYTHONPATH}"
          ${CUDAQ_PULSE_SPHINX_EXECUTABLE} -W --keep-going -b html
          ${CUDAQ_PULSE_SOURCE_DIR}/docs
          ${CUDAQ_PULSE_BINARY_DIR}/docs/html
  DEPENDS _cudaq_pulse_native
  COMMENT "Building CUDA-Q pulse documentation"
  USES_TERMINAL)
