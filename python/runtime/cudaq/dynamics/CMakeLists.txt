# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
install(TARGETS nvqir_dynamics_bindings DESTINATION cudaq/operator)

