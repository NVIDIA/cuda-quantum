# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_ionq 
  SOURCES IonQTester.cpp
  BACKEND ionq)


configure_file("IonQStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/ionq/IonQStartServerAndTest.sh" @ONLY)
add_test(NAME ionq-tests COMMAND bash IonQStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/ionq/)
