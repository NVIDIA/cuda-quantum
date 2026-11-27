# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.23)
project(cudaq-edem-parent LANGUAGES C CXX)
enable_testing()

# Exercise a parent that already defines GF2 and uses CUDA-Q's split component.
set(CMAKE_INSTALL_LIBDIR lib)
set(CUDAQ_CORE_LIBRARY_COMPONENT Core)
set(CUDAQ_CORE_DEVELOPMENT_COMPONENT Core)
set(CUDAQ_EDEM_BUILD_TESTS OFF)
add_subdirectory("${CUDAQ_EDEM_TEST_SOURCE}/../math/gf2" gf2)
add_subdirectory("${CUDAQ_EDEM_TEST_SOURCE}" edem)
add_subdirectory("${CUDAQ_EDEM_TEST_SOURCE}/test/consumer" consumer)
