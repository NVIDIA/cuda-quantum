# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-em-qir)

add_library(${LIBRARY_NAME} SHARED QubitQIRExecutionManager.cpp)
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>
    PRIVATE .)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC cudaq-spin PRIVATE nvqir cudaq-common fmt::fmt-header-only)

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-em-qir-targets DESTINATION lib)

install(EXPORT cudaq-em-qir-targets
        FILE CUDAQEmQirTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
