# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.23)
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
  CUDAQ_GF2_STANDALONE_BUILD)
if(CUDAQ_GF2_STANDALONE_BUILD)
  project(cudaq-gf2 LANGUAGES C CXX)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
endif()
set(CUDAQ_GF2_TESTS_DEFAULT "${CUDAQ_GF2_STANDALONE_BUILD}")
if(DEFINED CUDAQ_EDEM_BUILD_TESTS)
  set(CUDAQ_GF2_TESTS_DEFAULT "${CUDAQ_EDEM_BUILD_TESTS}")
endif()
option(CUDAQ_GF2_BUILD_TESTS "Build GF2 component tests" ${CUDAQ_GF2_TESTS_DEFAULT})

enable_language(CXX)
add_library(cudaq-gf2 lib/gf2.cpp lib/map.cpp lib/memory.cpp)
add_library(cudaq::gf2 ALIAS cudaq-gf2)

target_sources(cudaq-gf2 PUBLIC
  FILE_SET HEADERS
  BASE_DIRS include
  FILES include/cudaq/gf2/core.h include/cudaq/gf2/map.h
    include/cudaq/gf2/format.h include/cudaq/gf2/memory.h
    include/cudaq/gf2/bump_allocator.h
)
target_compile_features(cudaq-gf2 PUBLIC c_std_11)
target_compile_features(cudaq-gf2 PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # libstdc++ assertion handlers introduce a C++ runtime dependency. Undefine
  # inherited assertions and prevent GCC 15+ from enabling them at -O0.
  target_compile_definitions(cudaq-gf2 PRIVATE _GLIBCXX_NO_ASSERTIONS)
  target_compile_options(cudaq-gf2 PRIVATE
    -fno-exceptions -fno-rtti -U_GLIBCXX_ASSERTIONS)
endif()
set_target_properties(cudaq-gf2 PROPERTIES
  EXPORT_NAME gf2
  C_EXTENSIONS OFF
  CXX_EXTENSIONS OFF
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LINKER_LANGUAGE C
  POSITION_INDEPENDENT_CODE ON
)

if(CUDAQ_GF2_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
