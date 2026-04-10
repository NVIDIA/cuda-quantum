# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

pybind11_add_module(nvqir_dynamics_bindings pyDynamics.cpp)
target_include_directories(nvqir_dynamics_bindings PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)
target_include_directories(nvqir_dynamics_bindings 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/runtime
        ${CMAKE_SOURCE_DIR}/runtime/nvqir/cudensitymat
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(nvqir_dynamics_bindings PRIVATE
    cudaq-logger
    nvqir-dynamics
    fmt::fmt-header-only
)

if(APPLE)
  set(_origin_prefix "@loader_path")
else()
  set(_origin_prefix "$ORIGIN")
endif()

if(NOT SKBUILD)
  set_target_properties(nvqir_dynamics_bindings PROPERTIES
    INSTALL_RPATH "${_origin_prefix}/../../lib;${_origin_prefix}/../../lib/plugins"
    BUILD_RPATH   "${CMAKE_BINARY_DIR}/lib"
  )
else()
  set_target_properties(nvqir_dynamics_bindings PROPERTIES
    INSTALL_RPATH "${_origin_prefix}/../../lib;${_origin_prefix}/../../cuda_quantum.libs"
    BUILD_RPATH   "${CMAKE_BINARY_DIR}/lib"
  )
endif()

# Set output directory for ctest-based python test invocation, which uses cudaq python from the build directory.
set_target_properties(nvqir_dynamics_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python/cudaq/dynamics)
install(TARGETS nvqir_dynamics_bindings DESTINATION cudaq/dynamics)

