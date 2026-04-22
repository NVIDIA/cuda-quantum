# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-logger
  SHARED
    logger.cpp
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-logger)

target_include_directories(cudaq-logger
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
    $<INSTALL_INTERFACE:include>
  PRIVATE 
    fmt::fmt-header-only
)
target_link_libraries(cudaq-logger
  PRIVATE
    spdlog::spdlog
)

install(TARGETS cudaq-logger EXPORT cudaq-logger-targets DESTINATION lib)
install(EXPORT cudaq-logger-targets
        FILE CUDAQLoggerTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
