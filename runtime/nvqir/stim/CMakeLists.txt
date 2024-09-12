# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME nvqir-stim)
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(STIM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/tpls/Stim)
set(STIM_BINARY_DIR ${CMAKE_BINARY_DIR}/tpls/Stim)

# The EXCLUDE_FROM_ALL makes it so that only libstim is built. If other targets
# are desired (like the command-line tool), remove EXCLUDE_FROM_ALL below.
add_subdirectory(${STIM_SOURCE_DIR} ${STIM_BINARY_DIR} EXCLUDE_FROM_ALL)

add_library(${LIBRARY_NAME} SHARED StimCircuitSimulator.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

set (STIM_DEPENDENCIES libstim fmt::fmt-header-only cudaq-common)

# If -Wall is enabled (as is done in parent directories), Stim will not compile.
# So override that here.
target_compile_options(libstim PRIVATE -Wno-all)

target_include_directories(${LIBRARY_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME}
  PRIVATE ${STIM_DEPENDENCIES})

set_target_properties(${LIBRARY_NAME}
    PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib")

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_target_config(stim)
