# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(_custatevec_hints
  "${cuStateVec_ROOT}"
  "${CUSTATEVEC_ROOT}"
  "$ENV{cuStateVec_ROOT}"
  "$ENV{CUSTATEVEC_ROOT}"
  "$ENV{CUQUANTUM_INSTALL_PREFIX}"
  "$ENV{CUDA_PATH}"
  "/usr/local"
  "/usr/local/cuda"
  "/usr"
)

if(NOT DEFINED CUDAToolkit_VERSION)
  find_package(CUDAToolkit REQUIRED)
endif()

find_path(cuStateVec_INCLUDE_DIR
  NAMES custatevec.h
  HINTS ${_custatevec_hints}
  PATH_SUFFIXES include
)

find_library(cuStateVec_LIBRARY
  NAMES custatevec libcustatevec.so.1
  HINTS ${_custatevec_hints}
  PATH_SUFFIXES lib lib/${CUDAToolkit_VERSION_MAJOR} lib64 lib64/${CUDAToolkit_VERSION_MAJOR}
)

if(cuStateVec_INCLUDE_DIR AND EXISTS "${cuStateVec_INCLUDE_DIR}/custatevec.h")
  file(READ "${cuStateVec_INCLUDE_DIR}/custatevec.h" _cv_hdr)
  string(REGEX MATCH "#define CUSTATEVEC_VER_MAJOR[ \t]+([0-9]+)" _m "${_cv_hdr}")
  set(CUSTATEVEC_VER_MAJOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define CUSTATEVEC_VER_MINOR[ \t]+([0-9]+)" _m "${_cv_hdr}")
  set(CUSTATEVEC_VER_MINOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define CUSTATEVEC_VER_PATCH[ \t]+([0-9]+)" _m "${_cv_hdr}")
  set(CUSTATEVEC_VER_PATCH "${CMAKE_MATCH_1}")
  set(cuStateVec_VERSION "${CUSTATEVEC_VER_MAJOR}.${CUSTATEVEC_VER_MINOR}.${CUSTATEVEC_VER_PATCH}")

  set(_cusv_min_cuda "11")
  set(_cusv_max_cuda "13")
  if(CUDAToolkit_VERSION_MAJOR VERSION_LESS _cusv_min_cuda OR
     CUDAToolkit_VERSION_MAJOR VERSION_GREATER _cusv_max_cuda)
    message(FATAL_ERROR
      "cuStateVec ${cuStateVec_VERSION} supports CUDA >= ${_cusv_min_cuda} "
      "and <= ${_cusv_max_cuda}, but found CUDA ${CUDAToolkit_VERSION}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuStateVec
  REQUIRED_VARS cuStateVec_INCLUDE_DIR cuStateVec_LIBRARY
  VERSION_VAR cuStateVec_VERSION
)

if(cuStateVec_FOUND AND NOT TARGET cuStateVec::cuStateVec)
  add_library(cuStateVec::cuStateVec UNKNOWN IMPORTED)
  set_target_properties(cuStateVec::cuStateVec PROPERTIES
    IMPORTED_LOCATION "${cuStateVec_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${cuStateVec_INCLUDE_DIR}"
  )
endif()
