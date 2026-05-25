# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(_cutensor_hints
  "${cuTensor_ROOT}"
  "${CUTENSOR_ROOT}"
  "$ENV{cuTensor_ROOT}"
  "$ENV{CUTENSOR_ROOT}"
  "$ENV{CUTENSOR_INSTALL_PREFIX}"
  "$ENV{CUDA_PATH}"
  "/usr/local"
  "/usr/local/cuda"
  "/usr"
)

if(NOT DEFINED CUDAToolkit_VERSION)
  find_package(CUDAToolkit REQUIRED)
endif()

find_path(cuTensor_INCLUDE_DIR
  NAMES cutensor.h
  HINTS ${_cutensor_hints}
  PATH_SUFFIXES include
)

find_library(cuTensor_LIBRARY
  NAMES cutensor libcutensor.so.2
  HINTS ${_cutensor_hints}
  PATH_SUFFIXES lib lib/${CUDAToolkit_VERSION_MAJOR} lib64 lib64/${CUDAToolkit_VERSION_MAJOR}
)

if(cuTensor_INCLUDE_DIR AND EXISTS "${cuTensor_INCLUDE_DIR}/cutensor.h")
  file(READ "${cuTensor_INCLUDE_DIR}/cutensor.h" _ct_hdr)
  string(REGEX MATCH "#define CUTENSOR_MAJOR[ \t]+([0-9]+)" _m "${_ct_hdr}")
  set(CUTENSOR_VER_MAJOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define CUTENSOR_MINOR[ \t]+([0-9]+)" _m "${_ct_hdr}")
  set(CUTENSOR_VER_MINOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define CUTENSOR_PATCH[ \t]+([0-9]+)" _m "${_ct_hdr}")
  set(CUTENSOR_VER_PATCH "${CMAKE_MATCH_1}")
  set(cuTensor_VERSION "${CUTENSOR_VER_MAJOR}.${CUTENSOR_VER_MINOR}.${CUTENSOR_VER_PATCH}")

  set(_cut_min_cuda "11")
  set(_cut_max_cuda "13")
  if(CUDAToolkit_VERSION_MAJOR VERSION_LESS _cut_min_cuda OR
     CUDAToolkit_VERSION_MAJOR VERSION_GREATER _cut_max_cuda)
    message(FATAL_ERROR
      "cuTensor ${cuTensor_VERSION} supports CUDA >= ${_cut_min_cuda} "
      "and <= ${_cut_max_cuda}, but found CUDA ${CUDAToolkit_VERSION}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuTensor
  REQUIRED_VARS cuTensor_INCLUDE_DIR cuTensor_LIBRARY
  VERSION_VAR cuTensor_VERSION
)

if(cuTensor_FOUND AND NOT TARGET cuTensor::cuTensor)
  add_library(cuTensor::cuTensor UNKNOWN IMPORTED)
  set_target_properties(cuTensor::cuTensor PROPERTIES
      IMPORTED_LOCATION "${cuTensor_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${cuTensor_INCLUDE_DIR}"
  )
endif()
