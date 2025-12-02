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

# ======= Phase 1: Old tests removed ==========================================
# All old tests using device_ptr API have been removed.
# New Daemon-based tests will be added in Phase 4 of the integration plan.
# ==============================================================================

message(STATUS "Unit tests: Phase 1 - old tests removed, new tests will be added in Phase 4")
message(STATUS "  See docs/integration_plan.md Phase 4 for upcoming test infrastructure")


