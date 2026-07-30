# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Library-mode QIS tests (host quantum types / parameterized custom ops).
# Compiled with the ambient g++ under CUDAQ_LIBRARY_MODE from the parent
# unittests/CMakeLists.txt. Kernel coverage that needs nvq++ lives under
# unittests/nvqpp/qis/.

add_executable(test_qis ${CUDAQ_GTEST_MAIN} QubitQISTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_qis PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
endif()
target_compile_definitions(test_qis PRIVATE -DNVQIR_BACKEND_NAME=qpp)
target_link_libraries(test_qis
  PRIVATE
  cudaq
  cudaq-em-default
  cudaq-platform-default
  cudaq-mlir-runtime
  nvqir
  nvqir-qpp
  gtest_main)
cudaq_gtest_discover_tests(test_qis PROPERTIES PROCESSORS ${CUDAQ_TEST_OMP_SLOTS}
                            DISCOVERY_TIMEOUT 120)
