# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Unzip utility based on libz.
# Minizip is an addon library, not included by default in the official libz distribution.
# Here, we try to find the minizip library (e.g., if using `install_prerequisites.sh` to build libz).
# If failed to locate minizip, download minizip (unzip functionality) and compile directly.  
add_library(unzip_util STATIC helpers/UnzipUtils.cpp)
target_link_libraries(unzip_util PRIVATE fmt::fmt-header-only)
find_package(PkgConfig)
# By default, Minizip has package config (.pc) file.
# If CMake can find PkgConfig, use it to find minizip
if (PkgConfig_FOUND) 
  pkg_check_modules(MINI_ZIP REQUIRED IMPORTED_TARGET minizip)
  target_link_libraries(unzip_util PRIVATE PkgConfig::MINI_ZIP)
else()
  # No PkgConfig, locate the lib manually
  find_library(MINI_ZIP_LIB NAMES minizip)
  get_filename_component(MINI_LIB_DIR ${MINI_ZIP_LIB} DIRECTORY)
  find_file(MINI_UNZIP_INC NAMES unzip.h 
    HINTS 
      ${MINI_LIB_DIR}/../include
      ${MINI_LIB_DIR}/../include/minizip
  )
  if (NOT MINI_ZIP_LIB OR NOT MINI_UNZIP_INC)
    message(STATUS "Minizip NOT found. Trying to build it from source...")
    # Using version 1.3
    set(ZLIB_VERSION v1.3)
    file(DOWNLOAD https://raw.githubusercontent.com/madler/zlib/${ZLIB_VERSION}/contrib/minizip/unzip.h ${CMAKE_CURRENT_BINARY_DIR}/unzip.h)
    file(DOWNLOAD https://raw.githubusercontent.com/madler/zlib/${ZLIB_VERSION}/contrib/minizip/unzip.c ${CMAKE_CURRENT_BINARY_DIR}/unzip.c)
    file(DOWNLOAD https://raw.githubusercontent.com/madler/zlib/${ZLIB_VERSION}/contrib/minizip/ioapi.h ${CMAKE_CURRENT_BINARY_DIR}/ioapi.h)
    file(DOWNLOAD https://raw.githubusercontent.com/madler/zlib/${ZLIB_VERSION}/contrib/minizip/ioapi.c ${CMAKE_CURRENT_BINARY_DIR}/ioapi.c)
    # Build the lib directly
    add_library(mini_unzip STATIC ${CMAKE_CURRENT_BINARY_DIR}/unzip.c ${CMAKE_CURRENT_BINARY_DIR}/ioapi.c)
    target_link_libraries(mini_unzip PRIVATE z)
    target_link_libraries(unzip_util PRIVATE mini_unzip)
    target_include_directories(unzip_util PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  else()
    message(STATUS "Minizip found: ${MINI_ZIP_LIB}/${MINI_UNZIP_INC}")
    target_link_libraries(unzip_util PRIVATE ${MINI_ZIP_LIB})
    get_filename_component(MINI_INCLUDE_DIR ${MINI_UNZIP_INC} DIRECTORY)
    target_include_directories(unzip_util PRIVATE ${MINI_INCLUDE_DIR})
  endif()
endif()

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

target_link_libraries(${LIBRARY_NAME} PRIVATE cudaq cudaq-mlir-runtime fmt::fmt-header-only unzip_util)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
add_subdirectory(helpers/server_impl)
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
    cudaq 
    cudaq-em-default 
    cudaq-mlir-runtime
    cudaq-platform-default
    nvqir
    fmt::fmt-header-only
)
target_link_options(rest-remote-platform-server PRIVATE -Wl,--no-as-needed)
install(TARGETS rest-remote-platform-server DESTINATION lib)
