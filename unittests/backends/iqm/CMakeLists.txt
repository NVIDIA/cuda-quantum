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
  LINK_LIBS ${backend_unittest_libs} gmock_main
)

configure_file("IQMStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/iqm/IQMStartServerAndTest.sh" @ONLY)
add_test(NAME iqm-tests COMMAND bash IQMStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/iqm/)
set_property(TEST iqm-tests PROPERTY SKIP_RETURN_CODE 77)
