# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)
set(LIBRARY_NAME cudaq-builder)

add_library(cudaq-builder SHARED kernel_builder.cpp QuakeValue.cpp kernels.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})
target_include_directories(cudaq-builder PUBLIC 
          $<INSTALL_INTERFACE:include> 
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)
target_link_libraries(cudaq-builder
  PRIVATE
    cudaq
    fmt::fmt-header-only
    nvqir
    cudaq-mlir-runtime
)

install(TARGETS cudaq-builder DESTINATION lib)
