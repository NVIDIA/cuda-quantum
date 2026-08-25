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

GMP is a hard requirement: ``libmpfr`` is built on ``libgmp`` and ``mpfr.h``
includes ``gmp.h``, so this module fails unless ``FindGMP`` succeeds too.

Components
^^^^^^^^^^

``Headers``
  Require ``mpfr.h`` (and, since it includes ``gmp.h``, the GMP headers) in
  addition to the library. CUDA-Q redistributes the shared library but not the
  MPFR headers, so consumers that compile against ``mpfr.h`` (for instance
  through ``cudaq/Synthesis/Math/Real.h``) must request this component and
  provide their own MPFR development files.

Imported targets
^^^^^^^^^^^^^^^^

``MPFR::mpfr``
  The MPFR shared library, linking ``GMP::gmp`` and carrying the ``mpfr.h``
  directory as a usage requirement when the headers were found.

Result variables
^^^^^^^^^^^^^^^^

``MPFR_FOUND``, ``MPFR_LIBRARY``, ``MPFR_INCLUDE_DIR``, ``MPFR_VERSION``.
#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_package(GMP QUIET)

find_path(MPFR_INCLUDE_DIR
  NAMES mpfr.h
  DOC "Directory containing mpfr.h")

block()
  # Never accept libmpfr.a: only the dynamically linked library may be
  # redistributed under the terms CUDA-Q relies on.
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
  find_library(MPFR_LIBRARY
    NAMES mpfr
    HINTS "${CUDAQ_LIBRARY_DIR}"
    DOC "MPFR shared library")
endblock()

unset(MPFR_VERSION)
if(EXISTS "${MPFR_INCLUDE_DIR}/mpfr.h")
  file(STRINGS "${MPFR_INCLUDE_DIR}/mpfr.h" _mpfr_define
    REGEX "^#define[ \t]+MPFR_VERSION_STRING[ \t]+\"[^\"]+\"")
  if(_mpfr_define MATCHES "\"([^\"]+)\"")
    set(MPFR_VERSION "${CMAKE_MATCH_1}")
  endif()
  unset(_mpfr_define)
  # Anything that includes mpfr.h also needs gmp.h.
  if(GMP_INCLUDE_DIR)
    set(MPFR_Headers_FOUND TRUE)
  endif()
endif()

find_package_handle_standard_args(MPFR
  REQUIRED_VARS MPFR_LIBRARY GMP_FOUND
  VERSION_VAR MPFR_VERSION
  HANDLE_COMPONENTS
  REASON_FAILURE_MESSAGE
  "Run scripts/install_prerequisites.sh to build MPFR from source, or install \
it with your package manager (libmpfr-dev on apt, mpfr on brew) and set \
MPFR_ROOT or the MPFR_INSTALL_PREFIX environment variable. MPFR also needs GMP, \
which FindGMP reports on separately.")

if(MPFR_FOUND)
  if(NOT TARGET MPFR::mpfr)
    add_library(MPFR::mpfr SHARED IMPORTED)
    set_property(TARGET MPFR::mpfr PROPERTY IMPORTED_LOCATION "${MPFR_LIBRARY}")
    set_property(TARGET MPFR::mpfr APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES GMP::gmp)
  endif()
  if(MPFR_INCLUDE_DIR)
    set_property(TARGET MPFR::mpfr PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES "${MPFR_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(MPFR_INCLUDE_DIR MPFR_LIBRARY)
