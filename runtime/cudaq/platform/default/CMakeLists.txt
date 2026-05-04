# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-default)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")

add_library(cudaq-qpu-default OBJECT DefaultQPU.cpp)
target_include_directories(cudaq-qpu-default INTERFACE .)
target_include_directories(cudaq-qpu-default 
  PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime> 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/internal/compiler/include>)

set(CUDAQ_DEFAULTPLATFORM_SRC
  DefaultQuantumPlatform.cpp
)

add_library(${LIBRARY_NAME} SHARED ${CUDAQ_DEFAULTPLATFORM_SRC})
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    pthread cudaq-em-default cudaq-operator cudaq-common cudaq-logger
  PRIVATE
     cudaq CUDAQTargetConfigUtil cudaq-qpu-default)

install(TARGETS ${LIBRARY_NAME}
  EXPORT cudaq-platform-default-targets DESTINATION lib)

install(EXPORT cudaq-platform-default-targets
        FILE CUDAQPlatformDefaultTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)

if (OPENSSL_FOUND AND CUDAQ_ENABLE_REST)
  add_subdirectory(rest)
endif()
  
add_target_config(opt-test)
add_target_config(circuit-opt-bench)

if (cuStateVec_FOUND)
  add_target_config(nvidia)
  add_target_config(nvidia-fp64)
endif()
