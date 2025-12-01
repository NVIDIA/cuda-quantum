# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_quantum_machines QuantumMachinesTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_quantum_machines PRIVATE -Wl,--no-as-needed)
endif()
target_compile_definitions(test_quantum_machines PRIVATE -DNVQIR_BACKEND_NAME=quantum_machines)
target_include_directories(test_quantum_machines PRIVATE ../..)
target_link_libraries(test_quantum_machines
  PRIVATE fmt::fmt-header-only 
  cudaq-common 
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-operator
  cudaq-platform-default 
  gtest_main)


configure_file("QuantumMachinesStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/quantum_machines/QuantumMachinesStartServerAndTest.sh" @ONLY)
add_test(NAME quantum_machines-tests COMMAND bash QuantumMachinesStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/quantum_machines/) 