# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Unzip utility based on libz.
# Minizip is an addon library, not included by default in the official libz distribution.
# Hence, we require libz installation via the `install_prerequisites.sh` script, which does install minizip.
add_library(unzip_util STATIC helpers/UnzipUtils.cpp)
target_link_libraries(unzip_util PRIVATE fmt::fmt-header-only)
target_include_directories(unzip_util PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)
find_package(PkgConfig)
# By default, Minizip has package config (.pc) file.
# If CMake can find PkgConfig, use it to find minizip
find_path(MINIZIP_PKG_CONFIG_DIR NAMES minizip.pc
  HINTS 
    ${ZLIB_ROOT}/lib/pkgconfig
    $ENV{ZLIB_INSTALL_PREFIX}/lib/pkgconfig
    ${ZLIB_INCLUDE_DIR}/../lib/pkgconfig
)
if (PkgConfig_FOUND AND MINIZIP_PKG_CONFIG_DIR) 
  set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${MINIZIP_PKG_CONFIG_DIR}")
  pkg_check_modules(MINI_ZIP IMPORTED_TARGET minizip)
  # Make sure that we link to minizip static library
  if (MINI_ZIP_FOUND)
    find_library(MINI_ZIP_LIB NAMES libminizip.a
      HINTS 
        ${MINI_ZIP_LIBRARY_DIRS}
    )
    target_link_libraries(unzip_util PRIVATE ${MINI_ZIP_LIB})
    target_include_directories(unzip_util PRIVATE ${ZLIB_INCLUDE_DIR} ${MINI_ZIP_INCLUDE_DIRS})  
  endif()
else()
  # No PkgConfig, locate the lib manually
  # Make sure that we find minizip static library
  find_library(MINI_ZIP_LIB NAMES libminizip.a
    HINTS 
      ${ZLIB_ROOT}
      $ENV{ZLIB_INSTALL_PREFIX}
      ${ZLIB_INCLUDE_DIR}/../lib
  )
  get_filename_component(MINI_LIB_DIR ${MINI_ZIP_LIB} DIRECTORY)
  find_file(MINI_UNZIP_INC NAMES unzip.h 
    HINTS 
      ${MINI_LIB_DIR}/../include
      ${MINI_LIB_DIR}/../include/minizip
  )
  if (MINI_ZIP_LIB AND MINI_UNZIP_INC)
    message(STATUS "Minizip found: ${MINI_ZIP_LIB} and ${MINI_UNZIP_INC}")
    target_link_libraries(unzip_util PRIVATE ${MINI_ZIP_LIB} ZLIB::ZLIB)
    get_filename_component(MINI_INCLUDE_DIR ${MINI_UNZIP_INC} DIRECTORY)
    target_include_directories(unzip_util PRIVATE ${MINI_INCLUDE_DIR} ${ZLIB_INCLUDE_DIR})  
    set(MINI_ZIP_FOUND TRUE)
  endif()
endif()

if (NOT MINI_ZIP_FOUND)
  message(FATAL_ERROR "Minizip from zLib NOT found. Please run the 'install_prerequisites.sh' script to install zLib with Minizip")
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

target_link_libraries(${LIBRARY_NAME} PRIVATE cudaq cudaq-mlir-runtime fmt::fmt-header-only unzip_util nvqir)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
add_subdirectory(helpers/server_impl)
add_library(rest-remote-platform-server SHARED RemoteRuntimeServer.cpp helpers/RestRemoteServer.cpp helpers/GPUInfo.cpp)
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

if (CUDA_FOUND AND CUSTATEVEC_ROOT)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(rest-remote-platform-server PRIVATE CUDAQ_ENABLE_CUDA)
  target_link_libraries(rest-remote-platform-server PRIVATE CUDA::cudart_static)
endif()

target_link_options(rest-remote-platform-server PRIVATE -Wl,--no-as-needed)
install(TARGETS rest-remote-platform-server DESTINATION lib)
