# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_chrome_tracer ChromeTracerTester.cpp)

target_link_libraries(test_chrome_tracer
  PRIVATE
  cudaq-logger
  gtest_main)

gtest_discover_tests(test_chrome_tracer
  PROPERTIES ENVIRONMENT "CUDAQ_TIMING_TAGS=1"
  DISCOVERY_TIMEOUT 120)

add_executable(test_cudaq_fmt_format CudaqFmtFormatTester.cpp)

target_link_libraries(test_cudaq_fmt_format
  PRIVATE
  cudaq-logger
  gtest_main)

gtest_discover_tests(test_cudaq_fmt_format DISCOVERY_TIMEOUT 120)

add_executable(test_cudaq_log_macros CudaqLogMacrosTester.cpp)

target_link_libraries(test_cudaq_log_macros
  PRIVATE
  cudaq-logger
  gtest_main)

gtest_discover_tests(test_cudaq_log_macros DISCOVERY_TIMEOUT 120)
