# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(nvqir-qpp-observe-test SHARED QPPObserveBackend.cpp)
set (QPP_TEST_DEPENDENCIES "")
list(APPEND QPP_TEST_DEPENDENCIES fmt::fmt-header-only cudaq-common cudaq-spin)
add_openmp_configurations(nvqir-qpp-observe-test QPP_TEST_DEPENDENCIES)

target_include_directories(nvqir-qpp-observe-test
  PUBLIC . ${CMAKE_SOURCE_DIR}/runtime 
           ${CMAKE_SOURCE_DIR}/runtime/common 
           ${CMAKE_SOURCE_DIR}/runtime/nvqir 
           ${CMAKE_SOURCE_DIR}/tpls/eigen
)

target_link_libraries(nvqir-qpp-observe-test
                      PUBLIC libqpp
                      PRIVATE ${QPP_TEST_DEPENDENCIES})

add_executable(test_observe_backend QppObserveTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_observe_backend PRIVATE -Wl,--no-as-needed)
endif()
target_compile_definitions(test_observe_backend PRIVATE -DNVQIR_BACKEND_NAME=qpp_observe)
target_include_directories(test_observe_backend PRIVATE ../..)
target_link_libraries(test_observe_backend
  PRIVATE fmt::fmt-header-only 
  cudaq-common 
  cudaq
  nvqir 
  nvqir-qpp-observe-test
  cudaq-spin 
  cudaq-platform-default 
  cudaq-em-default
  gtest_main)
gtest_discover_tests(test_observe_backend)
