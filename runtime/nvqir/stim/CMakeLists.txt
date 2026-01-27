# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME nvqir-stim)
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(STIM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/tpls/Stim)
set(STIM_BINARY_DIR ${CMAKE_BINARY_DIR}/tpls/Stim)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    # Constrain to AVX-2 to keep ourselves compatible with x86-64-v3.
    set(SIMD_WIDTH 256 CACHE INTERNAL "Pass SIMD width to Stim subproject")
endif()

# The EXCLUDE_FROM_ALL makes it so that only libstim is built. If other targets
# are desired (like the command-line tool), remove EXCLUDE_FROM_ALL below.
add_subdirectory(${STIM_SOURCE_DIR} ${STIM_BINARY_DIR} EXCLUDE_FROM_ALL)

add_library(${LIBRARY_NAME} SHARED StimCircuitSimulator.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

set (STIM_DEPENDENCIES libstim fmt::fmt-header-only cudaq-common)

# If -Wall is enabled (as is done in parent directories), Stim will not compile.
# So override that here.
target_compile_options(libstim PRIVATE -Wno-all)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/nvqir-stim.map "
{
  global:
    *;
  local:
    *4stim*;
};
")

target_include_directories(${LIBRARY_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME}
  PRIVATE ${STIM_DEPENDENCIES})

target_link_options(${LIBRARY_NAME} PRIVATE 
  "-Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/nvqir-stim.map")

# Add a post-build check to verify that no unwanted Stim symbols are exported.
# This is important to allow users to import both `cudaq` and `stim` into the same
# Python environment (even if they are using different versions of Stim).
add_custom_command(
  TARGET ${LIBRARY_NAME}
  POST_BUILD
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/verify_linkage.sh $<TARGET_FILE:${LIBRARY_NAME}>
  VERBATIM
  COMMENT "Verifying that no stim namespace symbols are exported in ${LIBRARY_NAME}..."
)

set_target_properties(${LIBRARY_NAME}
    PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib")

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_target_config(stim)
