# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_chrome_tracer ChromeTracerTester.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_chrome_tracer PRIVATE -Wl,--no-as-needed)
endif()

target_link_libraries(test_chrome_tracer
  PRIVATE
  cudaq-logger
  gtest_main)

gtest_discover_tests(test_chrome_tracer
  PROPERTIES ENVIRONMENT "CUDAQ_TIMING_TAGS=1"
  DISCOVERY_TIMEOUT 120)

add_executable(test_cudaq_fmt_format CudaqFmtFormatTester.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_cudaq_fmt_format PRIVATE -Wl,--no-as-needed)
endif()

target_link_libraries(test_cudaq_fmt_format
  PRIVATE
  cudaq-logger
  gtest_main)

gtest_discover_tests(test_cudaq_fmt_format DISCOVERY_TIMEOUT 120)

add_executable(test_cudaq_log_macros CudaqLogMacrosTester.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_cudaq_log_macros PRIVATE -Wl,--no-as-needed)
endif()

target_link_libraries(test_cudaq_log_macros
  PRIVATE
  cudaq-logger
  gtest_main)

gtest_discover_tests(test_cudaq_log_macros DISCOVERY_TIMEOUT 120)
