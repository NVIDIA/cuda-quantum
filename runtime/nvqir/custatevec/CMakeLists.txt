# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_ENABLE_EXPORTS TRUE)
message (STATUS "CUSTATEVEC_ROOT and CUDA_FOUND - building custatevec NVQIR backend.")

macro (nvqir_create_cusv_plugin LIBRARY_NAME SOURCE_FILENAME) 
  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILENAME})

  target_include_directories(${LIBRARY_NAME}
    PRIVATE .. ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/runtime/common
    ${CUSTATEVEC_ROOT}/include)

  set (CUSTATEVEC_LIBDIR lib)
  if (EXISTS ${CUSTATEVEC_ROOT}/lib64)
    set (CUSTATEVEC_LIBDIR lib64)
  endif()

  target_link_libraries(${LIBRARY_NAME} PRIVATE fmt::fmt-header-only cudaq-common
                ${CUSTATEVEC_ROOT}/${CUSTATEVEC_LIBDIR}/libcustatevec_static.a
                ${CUDA_LIBRARIES} cublas )

  cudaq_library_set_rpath(${LIBRARY_NAME})
  
  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

endmacro()

nvqir_create_cusv_plugin(nvqir-custatevec CuStateVecCircuitSimulator.cu)
nvqir_create_cusv_plugin(nvqir-custatevec-f32 CuStateVecCircuitSimulatorF32.cu)
