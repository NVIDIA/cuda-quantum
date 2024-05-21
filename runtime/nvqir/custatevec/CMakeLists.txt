# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)
message (STATUS "CUSTATEVEC_ROOT and CUDA_FOUND - building custatevec NVQIR backend.")

add_library(nvqir-custatevec-kernels STATIC CuStateVecCircuitSimulator.cu)

string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
list(GET CUDA_VERSION_LIST 0 CUDA_MAJOR_VERSION)

find_library(CUSTATEVEC_LIB
    NAMES custatevec
    HINTS
    ${CUSTATEVEC_ROOT}/lib64
    ${CUSTATEVEC_ROOT}/lib
    ${CUSTATEVEC_ROOT}/lib64/${CUDA_MAJOR_VERSION}
    ${CUSTATEVEC_ROOT}/lib/${CUDA_MAJOR_VERSION})

get_filename_component(CUSTATEVEC_LIB_DIR ${CUSTATEVEC_LIB} DIRECTORY)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${CUSTATEVEC_LIB_DIR}")

macro (nvqir_create_cusv_plugin LIBRARY_NAME SOURCE_FILENAME)
  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILENAME})

  # Compiling from command line:
  # This command fails for older CUDA versions due to missing support for clang 16, 
  # but/and with CUDA 12.4 we get a very descriptive error 
  # "libc++ is not supported on x86 system" ...
  # It looks like this is generally the case, rather than a just not yet implemented.
  # The check can be bypassed with -D_ALLOW_UNSUPPORTED_LIBCPP, but that leads to errors.
  # /usr/local/cuda/bin/nvcc -allow-unsupported-compiler -ccbin=/usr/local/llvm/bin/clang++ ../runtime/nvqir/custatevec/CuStateVecCircuitSimulator.cu -I ../runtime/nvqir/ -I ../runtime/ -I ../tpls/spdlog/include/spdlog/ -I ../tpls/spdlog/include/ -I ../tpls/fmt/include/ -I ../runtime/common/ -I /usr/local/cuquantum/include/

  target_include_directories(${LIBRARY_NAME}
    PRIVATE 
      . .. 
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
      ${CMAKE_SOURCE_DIR}/runtime/common
      ${CUSTATEVEC_ROOT}/include)

  target_link_libraries(${LIBRARY_NAME} 
                        PRIVATE
                          fmt::fmt-header-only
                          cudaq-common
                          nvqir-custatevec-kernels
                          ${CUSTATEVEC_LIB}
                          ${CUDA_LIBRARIES}
                          cublas)
  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

endmacro()

nvqir_create_cusv_plugin(nvqir-custatevec-fp64 CuStateVecCircuitSimulator.cpp)
nvqir_create_cusv_plugin(nvqir-custatevec-fp32 CuStateVecCircuitSimulatorF32.cpp)
