# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(NVQLINK_QCS_SOURCES
  qcs_device.cpp
  control_server.cpp
)

add_library(nvqlink_qcs STATIC ${NVQLINK_QCS_SOURCES})

target_include_directories(nvqlink_qcs
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_NVQLINK_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(nvqlink_qcs
  PUBLIC
    nvqlink_network  # For RoCEChannel
    nvqlink_utils    # For logging
)

# QCS control plane requires nlohmann/json for UDP protocol
# Fetch it if not already available
include(FetchContent)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

target_link_libraries(nvqlink_qcs PUBLIC nlohmann_json::nlohmann_json)

