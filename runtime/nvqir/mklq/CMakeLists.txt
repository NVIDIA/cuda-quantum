# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(APPLE)
  enable_language(OBJCXX)
  find_library(MKLQ_METAL_FRAMEWORK Metal REQUIRED)
  find_library(MKLQ_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  set(MKLQ_METAL_RUNTIME_SOURCE MklqMetalRuntime.mm)
else()
  set(MKLQ_METAL_RUNTIME_SOURCE MklqMetalRuntimeFallback.cpp)
endif()

function(add_mklq_statevector_backend LIBRARY_NAME BACKEND_NAME CLASS_NAME
         PRINTED_NAME DIAGNOSTIC_NAME)
  add_library(${LIBRARY_NAME} SHARED MklqCpuCircuitSimulator.cpp ${ARGN})
  set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

  set(MKLQ_DEPENDENCIES fmt::fmt-header-only cudaq-common cudaq-logger)
  add_openmp_configurations(${LIBRARY_NAME} MKLQ_DEPENDENCIES)

  target_compile_definitions(${LIBRARY_NAME}
    PRIVATE
      MKLQ_SIMULATOR_BACKEND_NAME="${BACKEND_NAME}"
      MKLQ_SIMULATOR_CLASS=${CLASS_NAME}
      MKLQ_SIMULATOR_PRINTED_NAME=${PRINTED_NAME}
      MKLQ_SIMULATOR_DIAGNOSTIC_PREFIX="[${DIAGNOSTIC_NAME}]"
      MKLQ_SIMULATOR_STATE_DIAGNOSTIC_PREFIX="[${DIAGNOSTIC_NAME}-state]")

  target_include_directories(${LIBRARY_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      $<INSTALL_INTERFACE:include>)

  target_link_libraries(${LIBRARY_NAME}
    PRIVATE ${MKLQ_DEPENDENCIES})

  set_target_properties(${LIBRARY_NAME}
    PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib")

  if(APPLE)
    add_custom_command(TARGET ${LIBRARY_NAME} POST_BUILD
      COMMAND /usr/bin/codesign --force --sign - $<TARGET_FILE:${LIBRARY_NAME}>
      COMMENT "Ad-hoc signing ${LIBRARY_NAME} for macOS dlopen")
  endif()

  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
endfunction()

add_mklq_statevector_backend(
  nvqir-mklq_cpu
  mklq_cpu
  MklqCpuCircuitSimulator
  mklq_cpu
  mklq-cpu)

add_mklq_statevector_backend(
  nvqir-mklq_metal
  mklq_metal
  MklqMetalCircuitSimulator
  mklq_metal
  mklq-metal
  ${MKLQ_METAL_RUNTIME_SOURCE})
target_compile_definitions(nvqir-mklq_metal
  PRIVATE MKLQ_ENABLE_METAL_RUNTIME)

if(APPLE)
  target_link_libraries(nvqir-mklq_metal
    PRIVATE ${MKLQ_METAL_FRAMEWORK} ${MKLQ_FOUNDATION_FRAMEWORK})
endif()

add_target_config(mklq-cpu)
add_target_config(mklq-metal)
