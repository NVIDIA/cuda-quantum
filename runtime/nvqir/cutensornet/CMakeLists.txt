# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

message (STATUS "cuTensorNet_FOUND - building tensornet NVQIR backends.")
message(STATUS "Found cutensornet version: ${cuTensorNet_VERSION}")

# Ensure cuTensorNet and cuTensor library directories are on the install RPATH.
get_target_property(_cutn_loc cuTensorNet::cuTensorNet IMPORTED_LOCATION)
get_target_property(_cut_loc cuTensor::cuTensor IMPORTED_LOCATION)
if(_cutn_loc)
  get_filename_component(_cutn_dir "${_cutn_loc}" DIRECTORY)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_cutn_dir}")
endif()
if(_cut_loc)
  get_filename_component(_cut_dir "${_cut_loc}" DIRECTORY)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_cut_dir}")
endif()

# We need cutensornet v2.12.1+ 
# Note: before v2.12.1, there is a workaround for adjoint of non-unitary ops 
# which has been fixed in v2.12.1, so we've removed the workaround.
if (${cuTensorNet_VERSION} VERSION_GREATER_EQUAL "2.12.1")
  set (BASE_TENSOR_BACKEND_SRS tensornet_utils.cpp)

  # Helper macro to add cutensornet-based backends
  macro (nvqir_create_cutn_plugin LIBRARY_NAME CREATE_TARGET_CONFIG)
    # This will create a target named ${LIBRARY_NAME}
    add_library(nvqir-${LIBRARY_NAME} SHARED ${ARGN})
    target_include_directories(nvqir-${LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/runtime/common ${CMAKE_SOURCE_DIR}/runtime/nvqir ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(nvqir-${LIBRARY_NAME} PRIVATE fmt::fmt-header-only cudaq-logger cudaq cudaq-common cuTensorNet::cuTensorNet cuTensor::cuTensor CUDA::cudart_static)
    install(TARGETS nvqir-${LIBRARY_NAME} DESTINATION lib)
    if (${CREATE_TARGET_CONFIG})
      add_target_config(${LIBRARY_NAME})
    endif()
  endmacro()

  nvqir_create_cutn_plugin(tensornet TRUE ${BASE_TENSOR_BACKEND_SRS} simulator_tensornet_fp64_register.cpp)
  nvqir_create_cutn_plugin(tensornet-mps TRUE ${BASE_TENSOR_BACKEND_SRS} simulator_mps_fp64_register.cpp)
  nvqir_create_cutn_plugin(tensornet-fp32 FALSE ${BASE_TENSOR_BACKEND_SRS} simulator_tensornet_fp32_register.cpp)
  nvqir_create_cutn_plugin(tensornet-mps-fp32 FALSE ${BASE_TENSOR_BACKEND_SRS} simulator_mps_fp32_register.cpp)
  add_library(tensornet-mpi-util OBJECT mpi_support.cpp)
  target_include_directories(tensornet-mpi-util PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/runtime)
  target_link_libraries(tensornet-mpi-util PRIVATE cudaq-common cudaq-logger cuTensorNet::cuTensorNet)
  # Note: only tensornet backend supports MPI at cutensornet level (distributed tensor computation)
  target_link_libraries(nvqir-tensornet PRIVATE tensornet-mpi-util)
  target_link_libraries(nvqir-tensornet-fp32 PRIVATE tensornet-mpi-util)
else()
  message(WARNING "Skipped tensornet backend due to incompatible cutensornet version. Please install cutensornet v2.12.1+.")
endif()
