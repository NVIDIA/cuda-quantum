# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.24)
project(CoreConsumer LANGUAGES CXX)

# Configure against the installed core prefix, without the compiler SDK or
# frontend, to verify that its exported targets are independently consumable.
find_package(CUDAQ CONFIG REQUIRED)
add_executable(core-consumer main.cpp)
target_compile_features(core-consumer PRIVATE cxx_std_20)
target_include_directories(core-consumer PRIVATE "${CUDAQ_INCLUDE_DIR}")
target_link_libraries(core-consumer PRIVATE cudaq::cudaq-common)
enable_testing()
add_test(NAME core-consumer COMMAND core-consumer)
