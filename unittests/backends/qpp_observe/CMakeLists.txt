# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(nvqir-qpp-observe-test SHARED QPPObserveBackend.cpp)
set (QPP_TEST_DEPENDENCIES "")
list(APPEND QPP_TEST_DEPENDENCIES fmt::fmt-header-only cudaq-common cudaq-operator)
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

get_property(backend_unittest_libs DIRECTORY PROPERTY BACKEND_UNITTEST_LIBS)
add_backend_unittest_executable(test_observe_backend 
  SOURCES QppObserveTester.cpp
  BACKEND qpp_observe
  
  LINK_LIBS
  ${backend_unittest_libs}
  nvqir-qpp-observe-test
  cudaq-em-default
  )

gtest_discover_tests(test_observe_backend)
