# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[=======================================================================[.rst:
FindMPFR
--------

Find the GNU Multiple Precision Floating-Point Reliable Library.

Only the shared library is searched for: CUDA-Q links MPFR dynamically and
redistributes it under the LGPL v3, which requires that it remain replaceable.

This module is installed alongside ``CUDAQConfig.cmake``, which uses it to
re-resolve MPFR for consumers of the exported ``cudaq-synth`` target.

Result variables
^^^^^^^^^^^^^^^^

``MPFR_FOUND``, ``MPFR_LIBRARY``, ``MPFR_INCLUDE_DIR``, ``MPFR_VERSION``.
#]=======================================================================]

include(FindPackageHandleStandardArgs)

# CUDAQ_INCLUDE_DIR / CUDAQ_LIBRARY_DIR are set when this module is loaded from
# CUDAQConfig.cmake, and point at the MPFR the installation was built against.
# They are hints rather than roots so that MPFR_ROOT still wins in-tree.
find_path(MPFR_INCLUDE_DIR
  NAMES mpfr.h
  HINTS "${CUDAQ_INCLUDE_DIR}"
  DOC "Directory containing mpfr.h")

set(_mpfr_saved_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
find_library(MPFR_LIBRARY
  NAMES mpfr
  HINTS "${CUDAQ_LIBRARY_DIR}"
  DOC "MPFR shared library")
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_mpfr_saved_suffixes})
unset(_mpfr_saved_suffixes)

set(MPFR_VERSION "")
if(MPFR_INCLUDE_DIR AND EXISTS "${MPFR_INCLUDE_DIR}/mpfr.h")
  file(STRINGS "${MPFR_INCLUDE_DIR}/mpfr.h" _mpfr_define
    REGEX "^#define[ \t]+MPFR_VERSION_STRING[ \t]+\"[^\"]+\"")
  if(_mpfr_define MATCHES "\"([^\"]+)\"")
    set(MPFR_VERSION "${CMAKE_MATCH_1}")
  endif()
  unset(_mpfr_define)
endif()

if(MPFR_INCLUDE_DIR)
  set(MPFR_Headers_FOUND TRUE)
else()
  set(MPFR_Headers_FOUND FALSE)
endif()

find_package_handle_standard_args(MPFR
  REQUIRED_VARS MPFR_LIBRARY
  VERSION_VAR MPFR_VERSION
  HANDLE_COMPONENTS
  REASON_FAILURE_MESSAGE
  "Run scripts/install_prerequisites.sh to build MPFR from source, or install \
it with your package manager (libmpfr-dev on apt, mpfr on brew) and set \
MPFR_ROOT or the MPFR_INSTALL_PREFIX environment variable.")

if(MPFR_FOUND AND NOT TARGET MPFR::mpfr)
  # mpfr.h includes gmp.h and libmpfr is built on libgmp, so GMP is part of
  # MPFR's usage requirements.
  find_package(GMP QUIET)
  add_library(MPFR::mpfr UNKNOWN IMPORTED)
  set_target_properties(MPFR::mpfr PROPERTIES IMPORTED_LOCATION "${MPFR_LIBRARY}")
  if(MPFR_INCLUDE_DIR)
    set_target_properties(MPFR::mpfr PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MPFR_INCLUDE_DIR}")
  endif()
  if(TARGET GMP::gmp)
    set_property(TARGET MPFR::mpfr APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES GMP::gmp)
  endif()
endif()

mark_as_advanced(MPFR_INCLUDE_DIR MPFR_LIBRARY)
