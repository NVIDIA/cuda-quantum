# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
# # This source code and the accompanying materials are made available under #
# the terms of the Apache License 2.0 which accompanies this distribution. #
# ============================================================================ #

add_test(
  NAME cudaq-py-parallel-mqpu
  COMMAND ${PYTHON_EXECUTABLE} test_mqpu.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(
  cudaq-py-parallel-mqpu PROPERTIES ENVIRONMENT
                                    "PYTHONPATH=${CMAKE_BINARY_DIR}/python")


configure_file(run_mpi_py.sh.in "${CMAKE_CURRENT_BINARY_DIR}/run_mpi_py.sh" @ONLY)                                    
add_test(
  NAME cudaq-py-parallel-mpi
  COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2 bash run_mpi_py.sh
          ${CMAKE_SOURCE_DIR}/python/tests/parallel/test_mpi_mqpu.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(
  cudaq-py-parallel-mpi PROPERTIES ENVIRONMENT
                                   "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
