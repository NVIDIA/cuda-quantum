# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set_source_files_properties(TraceQPUSummaryPass.cpp
  PROPERTIES COMPILE_OPTIONS -fno-rtti)

set(plugin_root ${CMAKE_CURRENT_BINARY_DIR})
set(plugin_lib_dir ${plugin_root}/lib)
set(plugin_target_dir ${plugin_root}/targets)

file(MAKE_DIRECTORY ${plugin_lib_dir} ${plugin_target_dir})
configure_file(targets/trace_qpu.yml.in
               ${plugin_target_dir}/trace_qpu.yml
               @ONLY)

add_library(cudaq-example-qpu-trace_qpu EXCLUDE_FROM_ALL SHARED
  TraceQPU.cpp
  TraceQPUSummaryPass.cpp)
set_target_properties(cudaq-example-qpu-trace_qpu
  PROPERTIES
    OUTPUT_NAME cudaq-qpu-trace_qpu
    LIBRARY_OUTPUT_DIRECTORY ${plugin_lib_dir}
    RUNTIME_OUTPUT_DIRECTORY ${plugin_lib_dir}
    ARCHIVE_OUTPUT_DIRECTORY ${plugin_lib_dir})
target_include_directories(cudaq-example-qpu-trace_qpu
  PRIVATE ${PROJECT_SOURCE_DIR}/runtime)
target_link_libraries(cudaq-example-qpu-trace_qpu
  PRIVATE cudaq cudaq-common cudaq-logger MLIRFuncDialect MLIRIR MLIRPass)
