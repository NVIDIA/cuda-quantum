# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Compile-only: a C translation unit that includes the public headers and
# calls the header-only cudaq_dispatcher_get_device_handler.
add_library(cudaq-realtime-c-api-check OBJECT c_api_check.c)
set_target_properties(cudaq-realtime-c-api-check PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON)
target_include_directories(cudaq-realtime-c-api-check PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})
