# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
target_link_libraries(nvqir_dynamics_bindings PRIVATE nvqir-dynamics)
install(TARGETS nvqir_dynamics_bindings DESTINATION cudaq/operator)

