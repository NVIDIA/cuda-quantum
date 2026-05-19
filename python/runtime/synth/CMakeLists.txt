# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Python bindings for the Clifford+T rotation synthesis library
# (cudaq-synth). Skip silently if the cudaq-synth target was not built 
# (GMP/MPFR not found at configure time).
if(NOT TARGET cudaq-synth)
  return()
endif()

nanobind_add_module(_cudaq_synth
  NB_SHARED
  NB_DOMAIN cudaq
  py_synth.cpp)

target_include_directories(_cudaq_synth PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${nanobind_INCLUDE_DIR})

target_link_libraries(_cudaq_synth PRIVATE cudaq-synth)

if(APPLE)
  set(_origin_prefix "@loader_path")
else()
  set(_origin_prefix "$ORIGIN")
endif()

set_target_properties(_cudaq_synth PROPERTIES
  INSTALL_RPATH "${_origin_prefix}/../../lib"
  BUILD_RPATH   "${CMAKE_BINARY_DIR}/lib")

set_target_properties(_cudaq_synth PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/python/cudaq/synth)

install(TARGETS _cudaq_synth DESTINATION cudaq/synth)