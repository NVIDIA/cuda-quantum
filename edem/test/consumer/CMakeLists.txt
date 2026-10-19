# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.23)
project(cudaq-edem-consumer LANGUAGES C CXX)

if(NOT TARGET cudaq::edem)
  find_package(cudaq-edem ${CUDAQ_EDEM_TEST_VERSION} EXACT CONFIG REQUIRED
    PATHS "${CUDAQ_EDEM_TEST_PREFIX}" NO_DEFAULT_PATH)
endif()
if(NOT TARGET cudaq::gf2)
  message(FATAL_ERROR "The EDEM package must also export cudaq::gf2")
endif()

add_executable(edem_c_consumer main.c)
add_executable(edem_cpp_consumer main.cpp)
target_compile_features(edem_cpp_consumer PRIVATE cxx_std_17)
# Linking only EDEM must provide the GF2 dependency and its headers too.
target_link_libraries(edem_c_consumer PRIVATE cudaq::edem)
target_link_libraries(edem_cpp_consumer PRIVATE cudaq::edem)

enable_testing()
add_test(NAME edem_c_consumer COMMAND edem_c_consumer)
add_test(NAME edem_cpp_consumer COMMAND edem_cpp_consumer)
