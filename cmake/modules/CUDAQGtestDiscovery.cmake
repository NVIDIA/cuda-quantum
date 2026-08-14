# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Wrapper around gtest_discover_tests. When CUDAQ_TEST_SPLIT_GTESTS is ON
# (default everywhere except macOS), calls forward unchanged.
# When OFF, each call registers a single CTest test running the whole
# executable. On macOS every test process pays a multi-second dyld cost
# loading and validating the flat-namespace MLIR runtime libraries, so
# per-method registration multiplies that cost by the method count. See
# https://github.com/NVIDIA/cuda-quantum/issues/4857.

include_guard(GLOBAL)

include(GoogleTest)

if(APPLE)
  set(_CUDAQ_TEST_SPLIT_GTESTS_DEFAULT OFF)
else()
  set(_CUDAQ_TEST_SPLIT_GTESTS_DEFAULT ON)
endif()
option(CUDAQ_TEST_SPLIT_GTESTS
  "Register each gtest method as a separate CTest test. When OFF, register one CTest test per gtest executable to avoid repeated per-process startup costs."
  ${_CUDAQ_TEST_SPLIT_GTESTS_DEFAULT})

function(cudaq_gtest_discover_tests target)
  if(CUDAQ_TEST_SPLIT_GTESTS)
    gtest_discover_tests(${target} ${ARGN})
    return()
  endif()

  # Accepts the full gtest_discover_tests signature; discovery-only
  # arguments are ignored. Repeated PROPERTIES keywords accumulate.
  cmake_parse_arguments(ARG
    "NO_PRETTY_TYPES;NO_PRETTY_VALUES"
    "TEST_PREFIX;TEST_SUFFIX;WORKING_DIRECTORY;TEST_LIST;DISCOVERY_TIMEOUT;XML_OUTPUT_DIR;DISCOVERY_MODE"
    "EXTRA_ARGS;PROPERTIES;TEST_FILTER"
    ${ARGN})

  # TEST_PREFIX/TEST_SUFFIX keep names unique when the same executable is
  # registered more than once.
  set(test_name "${ARG_TEST_PREFIX}${target}${ARG_TEST_SUFFIX}")

  set(cmd_args ${ARG_EXTRA_ARGS})
  if(ARG_TEST_FILTER)
    list(APPEND cmd_args "--gtest_filter=${ARG_TEST_FILTER}")
  endif()

  if(ARG_WORKING_DIRECTORY)
    add_test(NAME ${test_name} COMMAND ${target} ${cmd_args}
             WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}")
  else()
    add_test(NAME ${test_name} COMMAND ${target} ${cmd_args})
  endif()

  if(ARG_PROPERTIES)
    set_tests_properties(${test_name} PROPERTIES ${ARG_PROPERTIES})
  endif()
endfunction()
