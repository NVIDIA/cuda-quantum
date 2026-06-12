# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(plugin_root ${CMAKE_CURRENT_BINARY_DIR})
set(plugin_lib_dir ${plugin_root}/lib)
set(plugin_target_dir ${plugin_root}/targets)

file(MAKE_DIRECTORY ${plugin_lib_dir} ${plugin_target_dir})
configure_file(targets/mock_rest.yml.in
               ${plugin_target_dir}/mock_rest.yml
               @ONLY)

add_library(cudaq-example-serverhelper-mock_rest EXCLUDE_FROM_ALL SHARED
  MockRestServerHelper.cpp)
set_target_properties(cudaq-example-serverhelper-mock_rest
  PROPERTIES
    OUTPUT_NAME cudaq-serverhelper-mock_rest
    LIBRARY_OUTPUT_DIRECTORY ${plugin_lib_dir}
    RUNTIME_OUTPUT_DIRECTORY ${plugin_lib_dir}
    ARCHIVE_OUTPUT_DIRECTORY ${plugin_lib_dir})
target_include_directories(cudaq-example-serverhelper-mock_rest
  PRIVATE ${PROJECT_SOURCE_DIR}/runtime)
target_link_libraries(cudaq-example-serverhelper-mock_rest
  PRIVATE cudaq-common cudaq-logger)
