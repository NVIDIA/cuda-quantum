# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Low-level utilities library.
# This library has no dependencies on other cudaq libraries, allowing both
# cudaq-operator and libcudaq to depend on it without circular dependencies.

set(LIBRARY_NAME cudaq-utils)

add_library(${LIBRARY_NAME} SHARED
  matrix.cpp
  cudaq_utils.cpp
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

target_include_directories(${LIBRARY_NAME}
    PUBLIC
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
       $<INSTALL_INTERFACE:include>
    PRIVATE .)

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-utils-targets DESTINATION lib)

install(EXPORT cudaq-utils-targets
        FILE CUDAQUtilsTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)

