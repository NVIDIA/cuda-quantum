# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[=======================================================================[.rst:
FindGMP
-------

Find the GNU Multiple Precision Arithmetic Library.

Only the shared library is searched for: CUDA-Q links GMP dynamically and
redistributes it under the LGPL v3, which requires that it remain replaceable.

This module is installed alongside ``CUDAQConfig.cmake``, which uses it to
re-resolve GMP for consumers of the exported ``cudaq-synth`` target.

Result variables
^^^^^^^^^^^^^^^^

``GMP_FOUND``, ``GMP_LIBRARY``, ``GMP_INCLUDE_DIR``, ``GMP_VERSION``.
#]=======================================================================]

include(FindPackageHandleStandardArgs)

# CUDAQ_INCLUDE_DIR / CUDAQ_LIBRARY_DIR are set when this module is loaded from
# CUDAQConfig.cmake, and point at the GMP the installation was built against.
# They are hints rather than roots so that GMP_ROOT still wins in-tree.
find_path(GMP_INCLUDE_DIR
  NAMES gmp.h
  HINTS "${CUDAQ_INCLUDE_DIR}"
  DOC "Directory containing gmp.h")

set(_gmp_saved_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
find_library(GMP_LIBRARY
  NAMES gmp
  HINTS "${CUDAQ_LIBRARY_DIR}"
  DOC "GMP shared library")
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_gmp_saved_suffixes})
unset(_gmp_saved_suffixes)

set(GMP_VERSION "")
if(GMP_INCLUDE_DIR AND EXISTS "${GMP_INCLUDE_DIR}/gmp.h")
  set(_gmp_version_parts)
  foreach(_gmp_macro IN ITEMS
      __GNU_MP_VERSION __GNU_MP_VERSION_MINOR __GNU_MP_VERSION_PATCHLEVEL)
    file(STRINGS "${GMP_INCLUDE_DIR}/gmp.h" _gmp_define
      REGEX "^#define[ \t]+${_gmp_macro}[ \t]+[0-9]+")
    if(_gmp_define MATCHES "[ \t]([0-9]+)[ \t]*$")
      list(APPEND _gmp_version_parts "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  list(JOIN _gmp_version_parts "." GMP_VERSION)
  unset(_gmp_version_parts)
  unset(_gmp_define)
endif()

if(GMP_INCLUDE_DIR)
  set(GMP_Headers_FOUND TRUE)
else()
  set(GMP_Headers_FOUND FALSE)
endif()

find_package_handle_standard_args(GMP
  REQUIRED_VARS GMP_LIBRARY
  VERSION_VAR GMP_VERSION
  HANDLE_COMPONENTS
  REASON_FAILURE_MESSAGE
  "Run scripts/install_prerequisites.sh to build GMP from source, or install \
it with your package manager (libgmp-dev on apt, gmp on brew) and set GMP_ROOT \
or the GMP_INSTALL_PREFIX environment variable.")

if(GMP_FOUND AND NOT TARGET GMP::gmp)
  add_library(GMP::gmp UNKNOWN IMPORTED)
  set_target_properties(GMP::gmp PROPERTIES IMPORTED_LOCATION "${GMP_LIBRARY}")
  if(GMP_INCLUDE_DIR)
    set_target_properties(GMP::gmp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARY)
