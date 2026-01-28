# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

configure_file(QuantinuumStartServerAndTest.sh.in QuantinuumStartServerAndTest.sh @ONLY)

# Mock server test
set(BACKEND_CONFIG "quantinuum emulate=false url=http://localhost:62440 credentials=FakeCppQuantinuum.config project=mock_project_id")

add_backend_unittest_executable(test_quantinuum 
  SOURCES QuantinuumTester.cpp
  BACKEND quantinuum
  BACKEND_CONFIG ${BACKEND_CONFIG}
)
add_test(NAME quantinuum-tests COMMAND bash QuantinuumStartServerAndTest.sh test_quantinuum WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Emulation test
string(REPLACE "emulate=false" "emulate=true" EMULATION_CONFIG ${BACKEND_CONFIG})
add_backend_unittest_executable(test_quantinuum_emulation 
  SOURCES QuantinuumEmulationTester.cpp 
  BACKEND quantinuum
  BACKEND_CONFIG ${EMULATION_CONFIG}
)
add_test(NAME quantinuum-emulation-tests COMMAND bash QuantinuumStartServerAndTest.sh test_quantinuum_emulation WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Machine=Helios-SC
string(CONCAT BACKEND_CONFIG ${BACKEND_CONFIG} " machine=Helios-SC")
add_backend_unittest_executable(test_quantinuum_ng
  SOURCES QuantinuumNGTester.cpp
  BACKEND quantinuum
  BACKEND_CONFIG ${BACKEND_CONFIG}
)
add_test(NAME quantinuum-tests-ng COMMAND bash QuantinuumStartServerAndTest.sh test_quantinuum_ng WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

 # Machine=Helios-SC + extra payload test
string(CONCAT BACKEND_CONFIG ${BACKEND_CONFIG} " extra_payload_provider=dummy")
add_backend_unittest_executable(test_quantinuum_payload
  SOURCES QuantinuumPayloadTester.cpp
  BACKEND quantinuum
  BACKEND_CONFIG ${BACKEND_CONFIG}
)
add_test(NAME quantinuum-tests-payload COMMAND bash QuantinuumStartServerAndTest.sh test_quantinuum_payload WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


# Testing nvqpp .cpp files
configure_file(QuantinuumStartServerAndTestNvqpp.sh.in QuantinuumStartServerAndTestNvqpp.sh @ONLY)
add_test(NAME quantinuum-tests-nvqpp COMMAND bash QuantinuumStartServerAndTestNvqpp.sh WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Tests are sharing the same port
set_tests_properties(
  quantinuum-tests
  quantinuum-emulation-tests
  quantinuum-tests-ng
  quantinuum-tests-payload
  quantinuum-tests-nvqpp
  PROPERTIES RESOURCE_LOCK "quantinuum_port")

# Test for Nexus helper functions
add_backend_unittest_executable(test_quantinuum_util 
  SOURCES QuantinuumUtilTester.cpp
  
  INCLUDES
  ../.. 
  ${CMAKE_SOURCE_DIR}/runtime/cudaq/platform/default/rest/helpers/quantinuum

  LINK_LIBS
  cudaq
  cudaq-common 
  cudaq-operator
  cudaq-platform-default
  gtest_main
)
gtest_discover_tests(test_quantinuum_util)
