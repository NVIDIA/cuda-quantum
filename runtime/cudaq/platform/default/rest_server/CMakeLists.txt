# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME rest-remote-platform-client)
add_library(${LIBRARY_NAME} SHARED 
                              helpers/RestRemoteClient.cpp
                              RemoteRuntimeClient.cpp 
)
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME} PRIVATE cudaq cudaq-mlir-runtime fmt::fmt-header-only)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
add_subdirectory(helpers/server_impl)
add_subdirectory(helpers/llvm_jit)
add_library(rest-remote-platform-server SHARED RemoteRuntimeServer.cpp helpers/RestRemoteServer.cpp)
target_include_directories(rest-remote-platform-server 
  PUBLIC 
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<INSTALL_INTERFACE:include>
  PRIVATE . ../../)
target_link_libraries(rest-remote-platform-server
  PRIVATE
    rest_server_impl
    jit_util
    cudaq 
    cudaq-em-default 
    cudaq-mlir-runtime
    cudaq-platform-default
    nvqir
    fmt::fmt-header-only
)
target_link_options(rest-remote-platform-server PRIVATE -Wl,--no-as-needed)
install(TARGETS rest-remote-platform-server DESTINATION lib)
