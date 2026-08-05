# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_path(GMP_INCLUDE_DIR NAMES gmp.h HINTS "${GMP_ROOT}/include")
find_path(MPFR_INCLUDE_DIR NAMES mpfr.h HINTS "${MPFR_ROOT}/include")

set(_synth_saved_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
find_library(GMP_LIBRARIES NAMES gmp libgmp HINTS "${GMP_ROOT}/lib")
find_library(MPFR_LIBRARIES NAMES mpfr libmpfr HINTS "${MPFR_ROOT}/lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_synth_saved_suffixes})

if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARIES OR
   NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARIES)
  message(FATAL_ERROR
    "The GMP/MPFR shared libraries were not found. They are required to "
    "build CUDA-Q (Clifford+T rotation synthesis). Run "
    "scripts/install_prerequisites.sh to build them from source, or install "
    "them via your package manager (e.g. libgmp-dev and libmpfr-dev on apt) "
    "and/or set GMP_INSTALL_PREFIX / MPFR_INSTALL_PREFIX.")
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

target_include_directories(cudaq-synth SYSTEM PUBLIC
  $<BUILD_INTERFACE:${GMP_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${MPFR_INCLUDE_DIR}>
)
target_include_directories(cudaq-synth PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

foreach(_synth_dep IN LISTS GMP_LIBRARIES MPFR_LIBRARIES)
  get_filename_component(_synth_dep_dir "${_synth_dep}" DIRECTORY)
  get_filename_component(_synth_dep_name "${_synth_dep}" NAME_WE)
  if(APPLE)
    file(GLOB _synth_dep_files
      "${_synth_dep_dir}/${_synth_dep_name}.dylib"
      "${_synth_dep_dir}/${_synth_dep_name}.*.dylib")
  else()
    file(GLOB _synth_dep_files "${_synth_dep_dir}/${_synth_dep_name}.so*")
  endif()
  install(FILES ${_synth_dep_files} DESTINATION lib)
endforeach()
