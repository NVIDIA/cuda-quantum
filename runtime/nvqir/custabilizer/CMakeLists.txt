# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(LIBRARY_NAME nvqir-custabilizer)
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

add_library(${LIBRARY_NAME} SHARED CustabilizerCircuitSimulator.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/nvqir-custabilizer.map "
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
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CUSTABILIZER_INCLUDE_DIR})

target_link_libraries(${LIBRARY_NAME}
  PRIVATE
    nvqir-stim
    libstim
    fmt::fmt-header-only
    cudaq-common
    ${CUSTABILIZER_LIBRARY}
    CUDA::cudart)

target_link_options(${LIBRARY_NAME}
  PRIVATE
    "-Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/nvqir-custabilizer.map")

get_filename_component(CUSTABILIZER_LIB_DIR ${CUSTABILIZER_LIBRARY} DIRECTORY)
set_target_properties(${LIBRARY_NAME}
  PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib:${CUSTABILIZER_LIB_DIR}")

add_custom_command(
  TARGET ${LIBRARY_NAME}
  POST_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/runtime/nvqir/stim/verify_linkage.sh $<TARGET_FILE:${LIBRARY_NAME}>
  VERBATIM
  COMMENT "Verifying that no stim namespace symbols are exported in ${LIBRARY_NAME}..."
)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_target_config(custabilizer)
