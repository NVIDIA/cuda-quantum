# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME nvqir)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(NVQIR_RUNTIME_SRC
  QIRTypes.cpp
  NVQIR.cpp
)

add_library(${LIBRARY_NAME} SHARED ${NVQIR_RUNTIME_SRC})
target_include_directories(${LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime> 
         $<INSTALL_INTERFACE:include> 
  PRIVATE .)

set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

# Private link to nvqir-qpp, it will always be here, 
# and by keeping it private, others can specify the
# backend library later on
target_link_libraries(${LIBRARY_NAME}
  PUBLIC cudaq-common cudaq-spin dl fmt::fmt-header-only)

install(TARGETS ${LIBRARY_NAME} 
        EXPORT nvqir-targets 
        LIBRARY DESTINATION lib
        INCLUDES DESTINATION include/nvqir)

add_subdirectory(qpp)

if (CUSTATEVEC_ROOT AND CUDA_FOUND) 
  add_subdirectory(custatevec)
endif() 
if (CUTENSORNET_ROOT AND CUDA_FOUND)
  add_subdirectory(cutensornet)  
endif()

install(EXPORT nvqir-targets
        FILE NVQIRTargets.cmake
        NAMESPACE nvqir::
        DESTINATION lib/cmake/nvqir)
