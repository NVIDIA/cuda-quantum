# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-spin)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(CUDAQ_SPIN_SRC
  spin_op.cpp matrix.cpp
)

add_library(${LIBRARY_NAME} SHARED ${CUDAQ_SPIN_SRC})
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
       $<INSTALL_INTERFACE:include>
    PRIVATE .)

set (SPIN_DEPENDENCIES "")
list(APPEND SPIN_DEPENDENCIES fmt::fmt-header-only)
add_openmp_configurations(${LIBRARY_NAME} SPIN_DEPENDENCIES)

target_link_libraries(${LIBRARY_NAME} PRIVATE ${SPIN_DEPENDENCIES})

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-spin-targets DESTINATION lib)

install(EXPORT cudaq-spin-targets
        FILE CUDAQSpinTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
