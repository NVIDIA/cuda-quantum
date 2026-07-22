# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Try to find MLIR from the cudaq-mlir wheel installation.
# This module sets:
#   CUDAQ_MLIR_FOUND
#   CUDAQ_MLIR_INCLUDE_DIRS
#   CUDAQ_MLIR_LIB_DIRS

if(NOT CUDAQ_MLIR_DIR)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
      "import mlir; import os; print(os.path.dirname(mlir.__file__))"
    OUTPUT_VARIABLE _MLIR_PY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_MLIR_PY_DIR)
    get_filename_component(CUDAQ_MLIR_DIR "${_MLIR_PY_DIR}/../.." ABSOLUTE)
  endif()
endif()

if(CUDAQ_MLIR_DIR)
  set(CUDAQ_MLIR_INCLUDE_DIRS "${CUDAQ_MLIR_DIR}/include")
  set(CUDAQ_MLIR_LIB_DIRS "${CUDAQ_MLIR_DIR}/lib")
  set(CUDAQ_MLIR_FOUND TRUE)
  message(STATUS "Found cudaq-mlir at ${CUDAQ_MLIR_DIR}")
else()
  set(CUDAQ_MLIR_FOUND FALSE)
  message(STATUS "cudaq-mlir wheel not found; MLIR_DIR must be set manually")
endif()
