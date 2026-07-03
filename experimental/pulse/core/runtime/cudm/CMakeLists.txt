# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(cudm_runtime SHARED
  cudm_runtime.cpp
)

target_include_directories(cudm_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(cudm_runtime PROPERTIES
  OUTPUT_NAME "cudm-runtime"
  VERSION 0.1.0
  SOVERSION 0
)

install(TARGETS cudm_runtime
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
