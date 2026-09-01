# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(GMP REQUIRED COMPONENTS Headers)
find_package(MPFR REQUIRED COMPONENTS Headers)

add_cudaq_library(cudaq-synth
  Circuit/Circuit.cpp
  Math/Diophantine.cpp
  Math/Geometry/Ellipse.cpp
  Math/Geometry/Rectangle.cpp
  Math/Geometry/ToUpright.cpp
  Math/Grid/Odgp.cpp
  Math/Grid/Tdgp.cpp
  Math/Real.cpp
  Synthesis/Gridsynth.cpp
  Synthesis/KmmSynthesize.cpp

  LINK_LIBS PUBLIC
  GMP::gmp
  MPFR::mpfr
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(cudaq-synth PUBLIC -Wno-psabi)
endif()

target_include_directories(cudaq-synth PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

foreach(_synth_dep IN ITEMS GMP::gmp MPFR::mpfr)
  get_target_property(_synth_dep_file ${_synth_dep} IMPORTED_LOCATION)
  get_filename_component(_synth_dep_dir "${_synth_dep_file}" DIRECTORY)
  get_filename_component(_synth_dep_name "${_synth_dep_file}" NAME_WE)
  if(APPLE)
    file(GLOB _synth_dep_files
      "${_synth_dep_dir}/${_synth_dep_name}.dylib"
      "${_synth_dep_dir}/${_synth_dep_name}.*.dylib")
  else()
    file(GLOB _synth_dep_files "${_synth_dep_dir}/${_synth_dep_name}.so*")
  endif()
  if(NOT _synth_dep_files)
    message(FATAL_ERROR "Found no shared library files to redistribute next to "
      "${_synth_dep_file}; CUDA-Q must ship ${_synth_dep} to comply with the "
      "LGPL v3.")
  endif()
  install(FILES ${_synth_dep_files} DESTINATION lib)
endforeach()
