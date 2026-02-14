# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_qudora 
  SOURCES QudoraTester.cpp
  BACKEND qudora
  BACKEND_CONFIG "qudora emulate=false url=http://localhost:62450/jobs")


configure_file(QudoraStartServerAndTest.sh.in QudoraStartServerAndTest.sh @ONLY)
add_test(NAME qudora-tests COMMAND bash QudoraStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
