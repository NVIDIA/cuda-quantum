# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_quantum_machines 
  SOURCES QuantumMachinesTester.cpp
  BACKEND quantum_machines
  )

configure_file("QuantumMachinesStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/quantum_machines/QuantumMachinesStartServerAndTest.sh" @ONLY)
add_test(NAME quantum_machines-tests COMMAND bash QuantumMachinesStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/quantum_machines/) 