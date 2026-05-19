# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Clifford+T rotation synthesis library

find_path(GMP_INCLUDE_DIR NAMES gmp.h)
find_path(MPFR_INCLUDE_DIR NAMES mpfr.h)

if(APPLE)
  set(_synth_saved_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  find_library(GMP_LIBRARIES NAMES gmp libgmp)
  find_library(MPFR_LIBRARIES NAMES mpfr libmpfr)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_synth_saved_suffixes})
else()
  find_library(GMP_LIBRARIES NAMES gmp libgmp)
  find_library(MPFR_LIBRARIES NAMES mpfr libmpfr)
endif()

if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARIES OR
   NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARIES)
  message(STATUS "cudaq-synth disabled: GMP or MPFR not found "
                 "(install libgmp-dev and libmpfr-dev to enable)")
  return()
endif()

set(CUDAQ_SYNTH_LOGGING_BACKEND "NONE" CACHE INTERNAL "")

add_library(cudaq-synth STATIC
  Math/Diophantine.cpp
  Math/Geometry/Ellipse.cpp
  Math/Geometry/Rectangle.cpp
  Math/Geometry/ToUpright.cpp
  Math/Grid/Odgp.cpp
  Math/Grid/Tdgp.cpp
  Math/Real.cpp
  Support/Logging.cpp
  Synthesis/Gridsynth.cpp
  Synthesis/KmmSynthesize.cpp
)

target_compile_features(cudaq-synth PUBLIC cxx_std_20)
set_target_properties(cudaq-synth PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(cudaq-synth PUBLIC -Wno-psabi)
endif()

target_include_directories(cudaq-synth
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GMP_INCLUDE_DIR}
    ${MPFR_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cudaq-synth
  PRIVATE
    ${GMP_LIBRARIES}
    ${MPFR_LIBRARIES}
)

set(CUDAQ_SYNTH_SIMD_BACKEND "AUTO" CACHE STRING
  "SIMD backend for BitVector: AUTO, AVX2, NEON, SCALAR")
set_property(CACHE CUDAQ_SYNTH_SIMD_BACKEND PROPERTY STRINGS
  "AUTO" "AVX2" "NEON" "SCALAR")

if(CUDAQ_SYNTH_SIMD_BACKEND STREQUAL "AVX2" OR
   (CUDAQ_SYNTH_SIMD_BACKEND STREQUAL "AUTO" AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64"))
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2" CUDAQ_SYNTH_HAS_AVX2)
  if(CUDAQ_SYNTH_HAS_AVX2)
    target_compile_options(cudaq-synth PRIVATE -mavx2)
    message(STATUS "cudaq-synth BitVector SIMD: AVX2")
  elseif(CUDAQ_SYNTH_SIMD_BACKEND STREQUAL "AVX2")
    message(WARNING
      "AVX2 requested but compiler does not support -mavx2; "
      "falling back to scalar")
  else()
    message(STATUS "cudaq-synth BitVector SIMD: scalar")
  endif()
elseif(CUDAQ_SYNTH_SIMD_BACKEND STREQUAL "NEON" OR
       (CUDAQ_SYNTH_SIMD_BACKEND STREQUAL "AUTO" AND
        CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm"))
  message(STATUS "cudaq-synth BitVector SIMD: NEON (compiler default)")
else()
  message(STATUS "cudaq-synth BitVector SIMD: scalar")
endif()
