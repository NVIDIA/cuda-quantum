# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-utils)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

# Create the CUDA-Q utils library
add_library(${LIBRARY_NAME} SHARED cudaq_utils.cpp matrix.cpp)

target_include_directories(${LIBRARY_NAME}
    PUBLIC $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
    PRIVATE .)

set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-utils-targets DESTINATION lib)

install(EXPORT cudaq-utils-targets
        FILE CUDAQUtilsTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
