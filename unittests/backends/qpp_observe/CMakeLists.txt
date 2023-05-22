# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(nvqir-qpp-observe-test SHARED QPPObserveBackend.cpp)
find_package(OpenMP)
set (QPP_DEPENDENCIES "")
if(OpenMP_CXX_FOUND)
message(STATUS "OpenMP Found. Adding build flags to QPP Backend: ${OpenMP_CXX_FLAGS}.")
list(APPEND QPP_DEPENDENCIES OpenMP::OpenMP_CXX)
  target_compile_definitions(nvqir-qpp-observe-test PRIVATE -DHAS_OPENMP=1)
endif()

target_include_directories(nvqir-qpp-observe-test
  PUBLIC . ${CMAKE_SOURCE_DIR}/runtime 
           ${CMAKE_SOURCE_DIR}/runtime/common 
           ${CMAKE_SOURCE_DIR}/runtime/nvqir 
           ${CMAKE_SOURCE_DIR}/tpls/eigen
           ${CMAKE_SOURCE_DIR}/tpls/qpp/include)
target_link_libraries(nvqir-qpp-observe-test
  PUBLIC fmt::fmt-header-only
          cudaq-common
          cudaq-spin ${QPP_DEPENDENCIES})

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
