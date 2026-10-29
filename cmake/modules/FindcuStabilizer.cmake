# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(_custabilizer_hints
  "${cuStabilizer_ROOT}"
  "${CUSTABILIZER_ROOT}"
  "$ENV{cuStabilizer_ROOT}"
  "$ENV{CUSTABILIZER_ROOT}"
  "$ENV{CUQUANTUM_INSTALL_PREFIX}"
  "$ENV{CUDA_PATH}"
  "/usr/local"
  "/usr/local/cuda"
  "/usr"
)

if(NOT DEFINED CUDAToolkit_VERSION)
  find_package(CUDAToolkit REQUIRED)
endif()

# Retain the existing cache variables for builds that specify cuStabilizer directly.
find_path(CUSTABILIZER_INCLUDE_DIR
  NAMES custabilizer.h
  HINTS ${_custabilizer_hints} ENV CUSTABILIZER_INCLUDE_DIR
  PATH_SUFFIXES include
)

find_library(CUSTABILIZER_LIBRARY
  NAMES custabilizer libcustabilizer.so.0
  HINTS ${_custabilizer_hints} ENV CUSTABILIZER_LIBRARY
  PATH_SUFFIXES lib lib/${CUDAToolkit_VERSION_MAJOR} lib64 lib64/${CUDAToolkit_VERSION_MAJOR} build/lib
)

if(CUSTABILIZER_INCLUDE_DIR AND EXISTS "${CUSTABILIZER_INCLUDE_DIR}/custabilizer.h")
  file(READ "${CUSTABILIZER_INCLUDE_DIR}/custabilizer.h" _cs_hdr)
  string(REGEX MATCH "#define CUSTABILIZER_MAJOR[ \t]+([0-9]+)" _m "${_cs_hdr}")
  set(CUSTABILIZER_VER_MAJOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define CUSTABILIZER_MINOR[ \t]+([0-9]+)" _m "${_cs_hdr}")
  set(CUSTABILIZER_VER_MINOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define CUSTABILIZER_PATCH[ \t]+([0-9]+)" _m "${_cs_hdr}")
  set(CUSTABILIZER_VER_PATCH "${CMAKE_MATCH_1}")
  set(cuStabilizer_VERSION "${CUSTABILIZER_VER_MAJOR}.${CUSTABILIZER_VER_MINOR}.${CUSTABILIZER_VER_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuStabilizer
  REQUIRED_VARS CUSTABILIZER_INCLUDE_DIR CUSTABILIZER_LIBRARY
  VERSION_VAR cuStabilizer_VERSION
)

if(cuStabilizer_FOUND AND NOT TARGET cuStabilizer::cuStabilizer)
  add_library(cuStabilizer::cuStabilizer UNKNOWN IMPORTED)
  set_target_properties(cuStabilizer::cuStabilizer PROPERTIES
    IMPORTED_LOCATION "${CUSTABILIZER_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUSTABILIZER_INCLUDE_DIR}"
  )
endif()
