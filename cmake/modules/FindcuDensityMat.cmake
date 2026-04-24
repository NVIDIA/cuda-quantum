# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(_cudensitymat_hints
  "${cuDensityMat_ROOT}"
  "${CUDENSITYMAT_ROOT}"
  "$ENV{cuDensityMat_ROOT}"
  "$ENV{CUDENSITYMAT_ROOT}"
  "$ENV{CUQUANTUM_INSTALL_PREFIX}"
  "$ENV{CUDA_PATH}"
  "/usr/local"
  "/usr/local/cuda"
  "/usr"
)

if(NOT DEFINED CUDAToolkit_VERSION)
  find_package(CUDAToolkit REQUIRED)
endif()

find_path(cuDensityMat_INCLUDE_DIR
  NAMES cudensitymat.h
  HINTS ${_cudensitymat_hints}
  PATH_SUFFIXES include
)

find_library(cuDensityMat_LIBRARY
  NAMES cudensitymat libcudensitymat.so.0
  HINTS ${_cudensitymat_hints}
  PATH_SUFFIXES lib lib/${CUDAToolkit_VERSION_MAJOR} lib64 lib64/${CUDAToolkit_VERSION_MAJOR}
)

if(cuDensityMat_INCLUDE_DIR AND EXISTS "${cuDensityMat_INCLUDE_DIR}/cudensitymat.h")
  file(READ "${cuDensityMat_INCLUDE_DIR}/cudensitymat.h" _cm_hdr)
  string(REGEX MATCH "CUDENSITYMAT_MAJOR ([0-9]*)" _ ${_cm_hdr})
  set(CUDENSITYMAT_MAJOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "CUDENSITYMAT_MINOR ([0-9]*)" _ ${_cm_hdr})
  set(CUDENSITYMAT_MINOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "CUDENSITYMAT_PATCH ([0-9]*)" _ ${_cm_hdr})
  set(CUDENSITYMAT_PATCH ${CMAKE_MATCH_1})
  set(cuDensityMat_VERSION ${CUDENSITYMAT_MAJOR}.${CUDENSITYMAT_MINOR}.${CUDENSITYMAT_PATCH})

  set(_cudm_min_cuda "11")
  set(_cudm_max_cuda "13")
  if(CUDAToolkit_VERSION_MAJOR VERSION_LESS _cudm_min_cuda OR
     CUDAToolkit_VERSION_MAJOR VERSION_GREATER _cudm_max_cuda)
    message(FATAL_ERROR
      "cuDensityMat ${cuDensityMat_VERSION} supports CUDA >= ${_cudm_min_cuda} "
      "and <= ${_cudm_max_cuda}, but found CUDA ${CUDAToolkit_VERSION}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDensityMat
  REQUIRED_VARS cuDensityMat_INCLUDE_DIR cuDensityMat_LIBRARY
  VERSION_VAR cuDensityMat_VERSION
)

if(cuDensityMat_FOUND AND NOT TARGET cuDensityMat::cuDensityMat)
  add_library(cuDensityMat::cuDensityMat UNKNOWN IMPORTED)
  set_target_properties(cuDensityMat::cuDensityMat PROPERTIES
    IMPORTED_LOCATION "${cuDensityMat_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${cuDensityMat_INCLUDE_DIR}"
  )
endif()
