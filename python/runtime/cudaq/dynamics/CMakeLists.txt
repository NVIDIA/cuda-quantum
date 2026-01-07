# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(nvqir_dynamics_bindings pyDynamics.cpp)
target_include_directories(nvqir_dynamics_bindings PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)
find_file(CUDENSITYMAT_INC
    NAMES   cudensitymat.h
    HINTS   
        $ENV{CUQUANTUM_INSTALL_PREFIX}/include      
        /usr/include    
        ENV CPATH
    REQUIRED
)

get_filename_component(CUDENSITYMAT_INCLUDE_DIR ${CUDENSITYMAT_INC} DIRECTORY)
target_include_directories(nvqir_dynamics_bindings 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/runtime
        ${CMAKE_SOURCE_DIR}/runtime/nvqir/cudensitymat 
        ${CUDENSITYMAT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(nvqir_dynamics_bindings PRIVATE fmt::fmt-header-only)
# Set output directory for ctest-based python test invocation, which uses cudaq python from the build directory. 
set_target_properties(nvqir_dynamics_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python/cudaq/dynamics)
install(TARGETS nvqir_dynamics_bindings DESTINATION cudaq/dynamics)

