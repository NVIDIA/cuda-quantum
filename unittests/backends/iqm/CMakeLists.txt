# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_property(backend_unittest_libs DIRECTORY PROPERTY BACKEND_UNITTEST_LIBS)
add_backend_unittest_executable(test_iqm 
  SOURCES IQMTester.cpp
  BACKEND iqm
  BACKEND_CONFIG "iqm emulate=false url=http://localhost:62443"
  LINK_LIBS ${backend_unittest_libs} gmock_main
)

configure_file(IQMStartServerAndTest.sh.in IQMStartServerAndTest.sh @ONLY)
add_test(NAME iqm-tests COMMAND bash IQMStartServerAndTest.sh test_iqm WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_property(TEST iqm-tests PROPERTY SKIP_RETURN_CODE 77)

set(dqa_filename "dqa_mock_qpu.txt")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${dqa_filename} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_backend_unittest_executable(test_iqm_dqa
  SOURCES IQMDqaTester.cpp
  BACKEND iqm
  BACKEND_CONFIG "iqm emulate=false url=http://localhost:62443 mapping_file=${dqa_filename}"
  LINK_LIBS ${backend_unittest_libs} gmock_main
)

add_test(NAME iqm-dqa-tests COMMAND bash IQMStartServerAndTest.sh test_iqm_dqa WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_property(TEST iqm-dqa-tests PROPERTY SKIP_RETURN_CODE 77)

# Tests are sharing the same port
set_tests_properties(iqm-tests iqm-dqa-tests PROPERTIES RESOURCE_LOCK "iqm_port")
