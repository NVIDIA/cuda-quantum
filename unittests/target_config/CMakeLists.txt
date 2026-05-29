# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_target_config TargetConfigTester.cpp)

# Tiny shared library used by the plugin-libraries test. Its constructor writes
# a sentinel file, giving the test a concrete signal that dlopen() happened.
add_library(cudaq-dlopen-sentinel-plugin SHARED DlopenSentinelPlugin.cpp)
set_target_properties(cudaq-dlopen-sentinel-plugin
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_target_config PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
endif()
target_link_libraries(test_target_config
  PRIVATE
    CUDAQTargetConfigUtil
    gtest_main
)

if(CUDAQ_ENABLE_PYTHON)
  target_include_directories(test_target_config PRIVATE
    ${CMAKE_SOURCE_DIR}/python/utils
  )
  # libcudaq.so has an unresolved reference to getRegisteredExecutionManager
  # that is normally provided at runtime by an execution manager plugin.
  # Link the default execution manager to satisfy it; this test does not
  # actually execute any kernels.
  target_link_libraries(test_target_config PRIVATE cudaq-py-utils cudaq-em-default)
  target_compile_definitions(test_target_config PRIVATE CUDAQ_ENABLE_PYTHON)
  target_compile_definitions(test_target_config
    PRIVATE
      "CUDAQ_DLOPEN_SENTINEL_PLUGIN_PATH=\"$<TARGET_FILE:cudaq-dlopen-sentinel-plugin>\""
      "CUDAQ_DLOPEN_SENTINEL_PLUGIN_FILENAME=\"$<TARGET_FILE_NAME:cudaq-dlopen-sentinel-plugin>\""
  )
  add_dependencies(test_target_config cudaq-dlopen-sentinel-plugin)
endif()

gtest_discover_tests(test_target_config DISCOVERY_TIMEOUT 120)
