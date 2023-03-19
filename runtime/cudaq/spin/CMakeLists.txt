# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-spin)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(CUDAQ_SPIN_SRC
  spin_op.cpp
)

add_library(${LIBRARY_NAME} SHARED ${CUDAQ_SPIN_SRC})
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>
    PRIVATE .)

cudaq_library_set_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-spin-targets DESTINATION lib)

install(EXPORT cudaq-spin-targets
        FILE CUDAQSpinTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
