# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)
message (STATUS "CUSTATEVEC_ROOT and CUDA_FOUND - building custatevec NVQIR backend.")

add_library(nvqir-custatevec-kernels SHARED CuStateVecCircuitSimulator.cu)
install(TARGETS nvqir-custatevec-kernels DESTINATION lib)

find_library(CUSTATEVEC_LIB
    NAMES custatevec
    HINTS
    ${CUSTATEVEC_ROOT}/lib64
    ${CUSTATEVEC_ROOT}/lib
    ${CUSTATEVEC_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
    ${CUSTATEVEC_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR})

get_filename_component(CUSTATEVEC_LIB_DIR ${CUSTATEVEC_LIB} DIRECTORY)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${CUSTATEVEC_LIB_DIR}")

macro (nvqir_create_cusv_plugin LIBRARY_NAME SOURCE_FILENAME)
  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILENAME})

  target_include_directories(${LIBRARY_NAME}
    PRIVATE 
      . .. 
      ${CUDAToolkit_INCLUDE_DIRS} 
      ${CMAKE_SOURCE_DIR}/runtime/common
      ${CUSTATEVEC_ROOT}/include)

  target_link_libraries(${LIBRARY_NAME} 
                        PRIVATE
                          fmt::fmt-header-only
                          cudaq-common
                          nvqir-custatevec-kernels
                          ${CUSTATEVEC_LIB}
                          CUDA::cudart_static
                          CUDA::cublas)
  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
endmacro()

nvqir_create_cusv_plugin(nvqir-custatevec-fp64 CuStateVecCircuitSimulator.cpp)
nvqir_create_cusv_plugin(nvqir-custatevec-fp32 CuStateVecCircuitSimulatorF32.cpp)
install(FILES CuStateVecCircuitSimulator.h DESTINATION include/nvqir)
