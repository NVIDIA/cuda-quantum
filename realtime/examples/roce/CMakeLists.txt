# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(NVQLINK_ENABLE_ROCE)
  include(FetchContent)
  FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )
  FetchContent_MakeAvailable(json)

  # UDP Control Server (shared by RoCE examples)
  add_library(udp_control_server STATIC udp_control_server.cpp)
  target_link_libraries(udp_control_server
    PUBLIC
      nvqlink_network
      nvqlink_daemon
      nvqlink_utils
      nlohmann_json::nlohmann_json
  )
  target_include_directories(udp_control_server 
    PUBLIC 
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CUDAQ_NVQLINK_INCLUDE_DIR}
  )
  
  # RoCE Daemon - Full RDMA daemon with UDP control
  add_executable(roce_daemon roce_daemon.cpp)
  target_link_libraries(roce_daemon
    PRIVATE
      nvqlink_daemon
      nvqlink_network
      nvqlink_utils
      udp_control_server
      nlohmann_json::nlohmann_json
  )
  
  # RoCE Channel - Worker threads with Channel API
  add_executable(roce_channel roce_channel.cpp)
  target_link_libraries(roce_channel
    PRIVATE
      nvqlink_daemon
      nvqlink_network
      nvqlink_utils
      udp_control_server
      nlohmann_json::nlohmann_json
  )
  
  # RoCE Daemon Client (single QP, function_id-based routing)
  add_executable(roce_daemon_client roce_daemon_client.cpp)
  target_link_libraries(roce_daemon_client
    PRIVATE
      ibverbs
      nlohmann_json::nlohmann_json
  )
  target_include_directories(roce_daemon_client 
    PRIVATE 
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
  # RoCE Channel Client (multi-QP, queue-based routing)
  add_executable(roce_channel_client roce_channel_client.cpp)
  target_link_libraries(roce_channel_client
    PRIVATE
      ibverbs
      nlohmann_json::nlohmann_json
  )
  target_include_directories(roce_channel_client 
    PRIVATE 
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
  message(STATUS "RoCE examples enabled:")
  message(STATUS "  - roce_daemon (full RDMA daemon)")
  message(STATUS "  - roce_channel (Channel API with workers)")
  message(STATUS "  - roce_daemon_client (client for Daemon mode)")
  message(STATUS "  - roce_channel_client (client for Channel mode)")
else()
  message(STATUS "RoCE examples disabled (requires -DNVQLINK_ENABLE_ROCE=ON)")
  message(STATUS "  Note: RoCE requires libibverbs library")
endif()
