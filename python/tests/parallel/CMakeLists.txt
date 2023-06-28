# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if (NOT MPI_CXX_FOUND)
  return()
endif()

if (NOT CUSTATEVEC_ROOT)
  return()
endif()

if (NOT CUDA_FOUND)
  return()
endif()

message(STATUS "MPI and CUDA Found, Adding PyMPITester")
configure_file("run_mpi_py.sh.in"
                    "${CMAKE_BINARY_DIR}/python/tests/parallel/run_mpi_py.sh" @ONLY)
add_test(NAME PyMPITest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 ${CMAKE_BINARY_DIR}/python/tests/parallel/run_mpi_py.sh)
set_tests_properties(PyMPITest PROPERTIES 
                    ENVIRONMENT 
                    "PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_BINARY_DIR}/python")  
add_test(NAME PyMQPUTest 
    COMMAND ${Python_EXECUTABLE} -m pytest -rP test_mqpu.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})