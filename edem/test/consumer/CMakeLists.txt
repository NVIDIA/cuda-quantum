# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.23)
option(CUDAQ_EDEM_TEST_C_ONLY "Consume EDEM without enabling C++" OFF)
if(CUDAQ_EDEM_TEST_C_ONLY)
  project(cudaq-edem-consumer LANGUAGES C)
else()
  project(cudaq-edem-consumer LANGUAGES C CXX)
endif()

if(NOT TARGET cudaq::edem)
  find_package(cudaq-edem ${CUDAQ_EDEM_TEST_VERSION} EXACT CONFIG REQUIRED
    PATHS "${CUDAQ_EDEM_TEST_PREFIX}" NO_DEFAULT_PATH)
endif()
if(NOT TARGET cudaq::gf2)
  message(FATAL_ERROR "The EDEM package must also export cudaq::gf2")
endif()

add_executable(edem_c_consumer main.c)
# Linking only EDEM must provide the GF2 dependency and its headers too.
target_link_libraries(edem_c_consumer PRIVATE cudaq::edem)

enable_testing()
add_test(NAME edem_c_consumer COMMAND edem_c_consumer)
if(NOT CUDAQ_EDEM_TEST_C_ONLY)
  add_executable(edem_cpp_consumer main.cpp)
  target_compile_features(edem_cpp_consumer PRIVATE cxx_std_17)
  target_link_libraries(edem_cpp_consumer PRIVATE cudaq::edem)
  add_test(NAME edem_cpp_consumer COMMAND edem_cpp_consumer)
  add_executable(edem_cpp_ownership_consumer ownership.cpp)
  set_target_properties(edem_cpp_ownership_consumer PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(edem_cpp_ownership_consumer PRIVATE
      -fno-exceptions -fno-rtti -Wall -Wextra -Wpedantic)
  endif()
  target_link_libraries(edem_cpp_ownership_consumer PRIVATE cudaq::edem)
  add_test(NAME edem_cpp_ownership_consumer COMMAND edem_cpp_ownership_consumer)
  add_executable(edem_composition_consumer composition.cpp)
  set_target_properties(edem_composition_consumer PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(edem_composition_consumer PRIVATE
      -fno-exceptions -fno-rtti -Wall -Wextra -Wpedantic)
  endif()
  target_link_libraries(edem_composition_consumer PRIVATE cudaq::edem)
  add_test(NAME edem_composition_consumer COMMAND edem_composition_consumer)
endif()
