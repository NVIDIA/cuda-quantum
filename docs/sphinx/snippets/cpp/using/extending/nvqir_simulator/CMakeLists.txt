/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

# [Begin Documentation]
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(DemoCreateNVQIRBackend VERSION 1.0.0 LANGUAGES CXX)

# CircuitSimulator.h uses C++20 features (concepts, requires-clauses,
# std::span), so this must be set explicitly -- linking against nvqir::nvqir
# does not force it on its own.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(NVQIR REQUIRED)

# Second argument is `GPURequirements`: `true` if this backend requires an
# NVIDIA GPU, `false` otherwise.
nvqir_add_backend(MySimulator false MySimulator.cpp)
# [End Documentation]