# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(LIBRARY_NAME nvqir-tracer)
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

add_library(${LIBRARY_NAME} SHARED Tracer.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

set (TRACER_DEPENDENCIES libstim cudaq-common)

target_include_directories(${LIBRARY_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME}
  PRIVATE ${TRACER_DEPENDENCIES})

# FIXME: try to refactor to avoid this public linking
target_link_libraries(${LIBRARY_NAME}
  PUBLIC fmt::fmt-header-only)

set_target_properties(${LIBRARY_NAME}
    PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib")

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_target_config(tracer)
