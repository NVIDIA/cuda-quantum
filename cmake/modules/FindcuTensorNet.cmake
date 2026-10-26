# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(_cutensornet_hints
  "${cuTensorNet_ROOT}"
  "${CUTENSORNET_ROOT}"
  "$ENV{cuTensorNet_ROOT}"
  "$ENV{CUTENSORNET_ROOT}"
  "$ENV{CUQUANTUM_INSTALL_PREFIX}"
  "$ENV{CUDA_PATH}"
  "/usr/local"
  "/usr/local/cuda"
  "/usr"
)

if(NOT DEFINED CUDAToolkit_VERSION)
  find_package(CUDAToolkit REQUIRED)
endif()

find_path(cuTensorNet_INCLUDE_DIR
  NAMES cutensornet.h
  HINTS ${_cutensornet_hints}
  PATH_SUFFIXES include
)

find_library(cuTensorNet_LIBRARY
  NAMES cutensornet libcutensornet.so.2
  HINTS ${_cutensornet_hints}
  PATH_SUFFIXES lib lib/${CUDAToolkit_VERSION_MAJOR} lib64 lib64/${CUDAToolkit_VERSION_MAJOR}
)

if(cuTensorNet_INCLUDE_DIR AND EXISTS "${cuTensorNet_INCLUDE_DIR}/cutensornet.h")
  file(READ "${cuTensorNet_INCLUDE_DIR}/cutensornet.h" _ct_hdr)
  string(REGEX MATCH "#define CUTENSORNET_MAJOR[ \t]+([0-9]+)" _m "${_ct_hdr}")
  set(CUTENSORNET_VER_MAJOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define CUTENSORNET_MINOR[ \t]+([0-9]+)" _m "${_ct_hdr}")
  set(CUTENSORNET_VER_MINOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define CUTENSORNET_PATCH[ \t]+([0-9]+)" _m "${_ct_hdr}")
  set(CUTENSORNET_VER_PATCH "${CMAKE_MATCH_1}")
  set(cuTensorNet_VERSION "${CUTENSORNET_VER_MAJOR}.${CUTENSORNET_VER_MINOR}.${CUTENSORNET_VER_PATCH}")

  set(_cutn_min_cuda "11")
  set(_cutn_max_cuda "13")
  if(CUDAToolkit_VERSION_MAJOR VERSION_LESS _cutn_min_cuda OR
     CUDAToolkit_VERSION_MAJOR VERSION_GREATER _cutn_max_cuda)
    message(FATAL_ERROR
      "cuTensorNet ${cuTensorNet_VERSION} supports CUDA >= ${_cutn_min_cuda} "
      "and <= ${_cutn_max_cuda}, but found CUDA ${CUDAToolkit_VERSION}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuTensorNet
  REQUIRED_VARS cuTensorNet_INCLUDE_DIR cuTensorNet_LIBRARY
  VERSION_VAR cuTensorNet_VERSION
)

if(cuTensorNet_FOUND AND NOT TARGET cuTensorNet::cuTensorNet)
  add_library(cuTensorNet::cuTensorNet UNKNOWN IMPORTED)
  set_target_properties(cuTensorNet::cuTensorNet PROPERTIES
      IMPORTED_LOCATION "${cuTensorNet_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${cuTensorNet_INCLUDE_DIR}"
  )
endif()
