# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

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

add_cudaq_library(cudaq-synth
  Math/Diophantine.cpp
  Math/Geometry/Ellipse.cpp
  Math/Geometry/Rectangle.cpp
  Math/Geometry/ToUpright.cpp
  Math/Grid/Odgp.cpp
  Math/Grid/Tdgp.cpp
  Math/Real.cpp
  Synthesis/Gridsynth.cpp
  Synthesis/KmmSynthesize.cpp

  LINK_LIBS PRIVATE
    ${GMP_LIBRARIES}
    ${MPFR_LIBRARIES}
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(cudaq-synth PUBLIC -Wno-psabi)
  if(TARGET obj.cudaq-synth)
    target_compile_options(obj.cudaq-synth PRIVATE -Wno-psabi)
  endif()
endif()

target_include_directories(cudaq-synth
  PUBLIC
    ${GMP_INCLUDE_DIR}
    ${MPFR_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
