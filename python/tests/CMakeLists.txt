# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests that generate MLIR and are run through both pytest and FileCheck.
add_subdirectory(compiler)
if (MPI_CXX_FOUND AND CUDA_FOUND)
  add_subdirectory(parallel)
endif()

if (NOT CUDAQ_TEST_MOCK_SERVERS)
  message(STATUS "CUDAQ_TEST_MOCK_SERVERS=FALSE, skipping Python remote QPU tests.")
  return()
endif()

execute_process(COMMAND ${Python_EXECUTABLE} -c "import fastapi"
  OUTPUT_VARIABLE FASTAPI_output
  ERROR_VARIABLE FASTAPI_error
  RESULT_VARIABLE FASTAPI_result)

execute_process(COMMAND ${Python_EXECUTABLE} -c "import pydantic"
  OUTPUT_VARIABLE PYDANTIC_output
  ERROR_VARIABLE PYDANTIC_error
  RESULT_VARIABLE PYDANTIC_result)

execute_process(COMMAND ${Python_EXECUTABLE} -c "import llvmlite"
  OUTPUT_VARIABLE LLVMLITE_output
  ERROR_VARIABLE LLVMLITE_error
  RESULT_VARIABLE LLVMLITE_result)

execute_process(COMMAND ${Python_EXECUTABLE} -c "import uvicorn"
  OUTPUT_VARIABLE UVICORN_output
  ERROR_VARIABLE UVICORN_error
  RESULT_VARIABLE UVICORN_result)

if (NOT ${FASTAPI_result} EQUAL 0)
  message(FATAL_ERROR "CUDA Quantum Python Warning - CUDAQ_TEST_MOCK_SERVERS=TRUE but fastapi module not found. Skipping Python remote QPU tests.")
  return()
endif()

if (NOT ${UVICORN_result} EQUAL 0)
  message(FATAL_ERROR "CUDA Quantum Python Warning - CUDAQ_TEST_MOCK_SERVERS=TRUE but uvicorn module not found. Skipping Python remote QPU tests.")
  return()
endif()

if (NOT ${LLVMLITE_result} EQUAL 0)
  message(FATAL_ERROR "CUDA Quantum Python Warning - CUDAQ_TEST_MOCK_SERVERS=TRUE but llvmlite module not found. Skipping Python remote QPU tests.")
  return()
endif()

if (NOT ${PYDANTIC_result} EQUAL 0)
  message(FATAL_ERROR "CUDA Quantum Python Warning - CUDAQ_TEST_MOCK_SERVERS=TRUE but pydantic module not found. Skipping Python remote QPU tests.")
  return()
endif()

message(STATUS "CUDAQ_TEST_MOCK_SERVERS=TRUE and dependent modules available, building Python remote QPU tests.")
