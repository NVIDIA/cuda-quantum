# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Minizip is a zlib addon installed by install_prerequisites.sh.
find_package(Minizip REQUIRED)
add_library(unzip_util STATIC helpers/UnzipUtils.cpp)
target_link_libraries(unzip_util PRIVATE fmt::fmt-header-only Minizip::Minizip)
target_include_directories(unzip_util PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)

set(LIBRARY_NAME rest-remote-platform-client)
add_library(${LIBRARY_NAME}
  SHARED
    helpers/RestRemoteClient.cpp
    RemoteRuntimeClient.cpp
)
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME} PRIVATE cudaq cudaq-logger cudaq-mlir-runtime fmt::fmt-header-only unzip_util nvqir)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
add_subdirectory(helpers/server_impl)
add_library(rest-remote-platform-server
  SHARED
    RemoteRuntimeServer.cpp
    helpers/RestRemoteServer.cpp
    helpers/GPUInfo.cpp
)

target_include_directories(rest-remote-platform-server 
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
    ../..
)

target_link_libraries(rest-remote-platform-server
  PRIVATE
    rest_server_impl
    cudaq 
    cudaq-operator
    cudaq-em-default 
    cudaq-mlir-runtime
    cudaq-platform-default
    cudaq-qir-verifier
    nvqir
    fmt::fmt-header-only
)

if (cuStateVec_FOUND)
  target_compile_definitions(rest-remote-platform-server PRIVATE CUDAQ_ENABLE_CUDA)
  target_link_libraries(rest-remote-platform-server PRIVATE CUDA::cudart_static)
endif()

target_link_options(rest-remote-platform-server PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
install(TARGETS rest-remote-platform-server DESTINATION lib)
