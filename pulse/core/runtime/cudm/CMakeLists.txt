# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudm_runtime SHARED
  cudm_runtime.cpp
)

# Use build/install-interface generator expressions: the raw source path is only
# valid while building in-tree, whereas consumers of the installed cudaq-pulse
# package resolve the header from the install prefix. A plain source path here is
# rejected by install(EXPORT) below.
target_include_directories(cudm_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudm_runtime PRIVATE
  cuDensityMat::cuDensityMat
  CUDA::cudart
  ${CMAKE_DL_LIBS})

get_target_property(_cudensitymat_location
  cuDensityMat::cuDensityMat IMPORTED_LOCATION)
if(_cudensitymat_location)
  get_filename_component(_cudensitymat_library_dir
    "${_cudensitymat_location}" DIRECTORY)
  set_target_properties(cudm_runtime PROPERTIES
    BUILD_RPATH "${_cudensitymat_library_dir}"
    INSTALL_RPATH "${_cudensitymat_library_dir}")
endif()

set_target_properties(cudm_runtime PROPERTIES
  OUTPUT_NAME "cudm-runtime"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  VERSION 0.1.0
  SOVERSION 0
)

# Join the cudaq-pulse export set so the installed package can expose
# cudaq::cudm_runtime via find_package(cudaq-pulse CONFIG). The public C ABI
# header ships alongside it so INSTALL_INTERFACE:include resolves for consumers.
install(TARGETS cudm_runtime
  EXPORT cudaq-pulse-targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(FILES cudm_runtime.h DESTINATION include)
