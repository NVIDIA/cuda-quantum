# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Mock server test
set(BACKEND_CONFIG "braket emulate=false url=http://localhost:62445 credentials=FakeCppBraket.config")

add_backend_unittest_executable(test_braket 
  SOURCES BraketTester.cpp
  BACKEND braket
  BACKEND_CONFIG ${BACKEND_CONFIG}
)

configure_file(BraketStartServerAndTest.sh.in BraketStartServerAndTest.sh @ONLY)
add_test(NAME braket-tests COMMAND bash BraketStartServerAndTest.sh test_braket WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Emulation test
string(REPLACE "emulate=false" "emulate=true" BACKEND_CONFIG ${BACKEND_CONFIG})
add_backend_unittest_executable(test_braket_emulation 
  SOURCES BraketEmulationTester.cpp 
  BACKEND braket
  BACKEND_CONFIG ${BACKEND_CONFIG}
)
add_test(NAME braket-emulation-tests COMMAND bash BraketStartServerAndTest.sh test_braket_emulation WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Tests are sharing the same port
set_tests_properties(braket-tests braket-emulation-tests PROPERTIES RESOURCE_LOCK "braket_port")
