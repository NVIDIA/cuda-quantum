# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

function(add_backend_test target )
  set(singleValues BACKEND_NAME)
  set(multiValues SOURCES)
  cmake_parse_arguments(ARG "" "${singleValues}" "${multiValues}" ${ARGN})

  add_executable(${target} ${ARG_SOURCES})
  target_compile_definitions(${target} PRIVATE NVQIR_BACKEND_NAME=${ARG_BACKEND_NAME})

  target_link_libraries(${target}
      PRIVATE
      nvqir
      nvqir-${ARG_BACKEND_NAME}
      cudaq
      cudaq-logger
      fmt::fmt-header-only
      gtest_main)
  gtest_discover_tests(${target} DISCOVERY_TIMEOUT 120)
endfunction()


add_backend_test(qpp_tester 
  BACKEND_NAME qpp
  SOURCES QPPTester.cpp)


add_backend_test(dm_tester 
  BACKEND_NAME dm
  SOURCES QPPDMTester.cpp)


add_backend_test(stim_tester 
  BACKEND_NAME stim
  SOURCES StimTester.cpp)
target_include_directories(stim_tester PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/nvqir/stim
  ${CMAKE_SOURCE_DIR}/tpls/Stim/src)
target_link_libraries(stim_tester PRIVATE libstim)
