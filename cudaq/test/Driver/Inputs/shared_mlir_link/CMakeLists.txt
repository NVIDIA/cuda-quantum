# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.23)
project(SharedMLIRLink LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MLIR_TARGETS)
  include("${MLIR_TARGETS}")
else()
  include("${CUDAQ_SOURCE_DIR}/cmake/modules/AddCUDAQ.cmake")
  add_library(obj.SharedRuntime OBJECT runtime.cpp)
  add_library(SharedRuntime INTERFACE)
  set(CUDAQ_MLIR_BUNDLED_LIBS SharedRuntime)
  set(CUDAQ_CORE_EXPORT_SET SharedMLIRTargets)
  set(CUDAQ_CORE_LIBRARY_COMPONENT Runtime)
  include("${CUDAQ_SOURCE_DIR}/cudaq/lib/Optimizer/cudaqMLIR-shlib.cmake")
  install(EXPORT SharedMLIRTargets NAMESPACE cudaq::
    DESTINATION lib/cmake/SharedMLIR)
endif()

add_library(static_copy STATIC runtime.cpp)
target_compile_definitions(static_copy PRIVATE MLIR_RUNTIME_VALUE=2)
add_library(transitive_mlir INTERFACE)
target_link_libraries(transitive_mlir INTERFACE cudaq::cudaqMLIR)

enable_testing()
foreach(kind IN ITEMS direct transitive)
  add_executable(${kind} consumer.cpp)
  if(kind STREQUAL direct)
    target_link_libraries(${kind} PRIVATE static_copy cudaq::cudaqMLIR)
  else()
    target_link_libraries(${kind} PRIVATE static_copy transitive_mlir)
  endif()
  add_test(NAME ${kind} COMMAND ${kind})
endforeach()
