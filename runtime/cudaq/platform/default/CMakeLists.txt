# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-default)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(CUDAQ_DEFAULTPLATFORM_SRC
  DefaultQuantumPlatform.cpp
  ../common/QuantumExecutionQueue.cpp
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
    pthread cudaq-em-default cudaq-spin cudaq-common
  PRIVATE
    fmt::fmt-header-only cudaq CUDAQTargetConfigUtil)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
install(TARGETS ${LIBRARY_NAME}
  EXPORT cudaq-platform-default-targets DESTINATION lib)

install(EXPORT cudaq-platform-default-targets
        FILE CUDAQPlatformDefaultTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)

if (OPENSSL_FOUND AND CUDAQ_ENABLE_REST)
  add_subdirectory(rest)
  add_subdirectory(rest_server)
endif()

if (CUSTATEVEC_ROOT AND CUDA_FOUND)
  add_target_config(nvidia)
  add_target_config(nvidia-fp64)
endif()
