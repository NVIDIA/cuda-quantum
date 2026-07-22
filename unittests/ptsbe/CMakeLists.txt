# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_ptsbe ${CUDAQ_GTEST_MAIN}
  KrausSelectionTester.cpp
  KrausTrajectoryTester.cpp
  ShotAllocationTester.cpp
  PTSBEInterfaceTester.cpp
  TraceConversionTester.cpp
  MergeTasksWithTrajectoryTester.cpp
  ExecutePTSBETester.cpp
  PTSBEPolicyTester.cpp
  PTSSamplingStrategyTester.cpp
  NoiseExtractorTester.cpp
  TrajectoryDeduplicationTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_ptsbe PRIVATE -Wl,--no-as-needed)
endif()
target_include_directories(test_ptsbe PRIVATE .)
target_link_libraries(test_ptsbe
  PRIVATE
  cudaq
  cudaq-em-default
  cudaq-platform-default
  cudaq-mlir-runtime
  nvqir
  nvqir-qpp
  fmt::fmt-header-only
  gtest_main)
cudaq_gtest_discover_tests(test_ptsbe PROPERTIES PROCESSORS ${CUDAQ_TEST_OMP_SLOTS} DISCOVERY_TIMEOUT 120)