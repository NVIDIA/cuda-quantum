# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies 
# ==============================================================================

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(googletest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PUBLIC --param=evrp-mode=legacy)
endif()
include(GoogleTest)


add_compile_options(-Wno-attributes) 

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test_config.h.in"
               "${CMAKE_BINARY_DIR}/include/test_config.h" 
               @ONLY)
include_directories(${CMAKE_BINARY_DIR}/include)

# ======= Phase 4: New Test Infrastructure ====================================

# Test utilities library (LoopbackChannel for testing without hardware)
add_library(test_utils STATIC
  test_utils/loopback_channel.cpp
)

target_include_directories(test_utils
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAQ_NVQLINK_INCLUDE_DIR}
)

target_link_libraries(test_utils
  PUBLIC
    nvqlink_network
)

# Daemon tests
add_executable(test_daemon test_daemon.cpp)
target_link_libraries(test_daemon 
  PRIVATE 
    GTest::gtest_main
    cudaq-nvqlink
    nvqlink_daemon
    test_utils
)
add_dependencies(NVQLINKUnitTests test_daemon)
gtest_discover_tests(test_daemon)

# Stream tests (serialization/deserialization)
add_executable(test_streams test_streams.cpp)
target_link_libraries(test_streams 
  PRIVATE 
    GTest::gtest_main
    nvqlink_network
)
add_dependencies(NVQLINKUnitTests test_streams)
gtest_discover_tests(test_streams)

message(STATUS "Unit tests: Phase 4 - New Daemon-based tests added")
message(STATUS "  - test_daemon (with LoopbackChannel)")
message(STATUS "  - test_streams (serialization)")

# ==============================================================================


