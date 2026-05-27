# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# DEM (Detector Error Model) analysis primitive.
add_library(cudaq-analysis SHARED Dem.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-analysis)

target_include_directories(cudaq-analysis
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
)

target_link_libraries(cudaq-analysis
  PUBLIC
    cudaq-common
  PRIVATE
    cudaq-logger
    cudaq-mlir-runtime-headers
    nvqir
    libstim
    fmt::fmt-header-only
)

install(TARGETS cudaq-analysis EXPORT cudaq-targets DESTINATION lib)
