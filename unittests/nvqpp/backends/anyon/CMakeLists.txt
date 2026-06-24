# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Mock server test
set(BACKEND_CONFIG "anyon emulate=false url=http://localhost:62446 credentials=FakeCppAnyon.config machine=telegraph-8q")

add_backend_unittest_executable(test_anyon 
  SOURCES AnyonTester.cpp 
  BACKEND anyon
  BACKEND_CONFIG ${BACKEND_CONFIG}
)

configure_file(AnyonStartServerAndTest.sh.in AnyonStartServerAndTest.sh @ONLY)
add_test(NAME anyon-tests COMMAND bash AnyonStartServerAndTest.sh test_anyon WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Emulation test
string(REPLACE "emulate=false" "emulate=true" BACKEND_CONFIG ${BACKEND_CONFIG})
add_backend_unittest_executable(test_anyon_emulation 
  SOURCES AnyonEmulationTester.cpp 
  BACKEND anyon
  BACKEND_CONFIG ${BACKEND_CONFIG}
)
add_test(NAME anyon-emulation-tests COMMAND bash AnyonStartServerAndTest.sh test_anyon_emulation WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Tests are sharing the same port
set_tests_properties(anyon-tests anyon-emulation-tests PROPERTIES RESOURCE_LOCK "anyon_port")