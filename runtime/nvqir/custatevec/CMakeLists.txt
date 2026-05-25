# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

message (STATUS "cuStateVec_FOUND - building custatevec NVQIR backend.")

# Ensure the cuStateVec library directory is on the install RPATH.
get_target_property(_cusv_loc cuStateVec::cuStateVec IMPORTED_LOCATION)
if(_cusv_loc)
  get_filename_component(_cusv_dir "${_cusv_loc}" DIRECTORY)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_cusv_dir}")
endif()

add_library(nvqir-custatevec-kernels SHARED CuStateVecCircuitSimulator.cu)
install(TARGETS nvqir-custatevec-kernels DESTINATION lib)

macro (nvqir_create_cusv_plugin LIBRARY_NAME SOURCE_FILENAME)
  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILENAME})

  target_include_directories(${LIBRARY_NAME}
    PRIVATE 
      . .. 
      ${CUDAToolkit_INCLUDE_DIRS} 
      ${CUDAToolkit_INCLUDE_DIRS}/cccl
      ${CMAKE_SOURCE_DIR}/runtime/common)

  # Prevent libc++ from defining the conflicting new/delete helpers
  target_compile_definitions(${LIBRARY_NAME} PRIVATE
    _LIBCPP_DISABLE_NEW_DELETE_DEFINITIONS
  )

  # Tell clang not to emit aligned new/delete operators
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${LIBRARY_NAME} PRIVATE -fno-aligned-allocation)
  endif()

  target_link_libraries(${LIBRARY_NAME} 
                        PRIVATE
                          cudaq-logger
                          cudaq-common
                          nvqir-custatevec-kernels
                          cuStateVec::cuStateVec
                          CUDA::cudart_static
                          CUDA::cublas)
  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
endmacro()

nvqir_create_cusv_plugin(nvqir-custatevec-fp64 CuStateVecCircuitSimulator.cpp)
nvqir_create_cusv_plugin(nvqir-custatevec-fp32 CuStateVecCircuitSimulatorF32.cpp)
install(FILES CuStateVecCircuitSimulator.h DESTINATION include/nvqir)
