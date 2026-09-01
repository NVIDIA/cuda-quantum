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

Components
^^^^^^^^^^

``Headers``
  Require ``gmp.h`` in addition to the library. CUDA-Q redistributes the shared
  library but not the GMP headers, so consumers that compile against ``gmp.h``
  (for instance through ``cudaq/Synthesis/Math/Integer.h``) must request this
  component and provide their own GMP development files.

Imported targets
^^^^^^^^^^^^^^^^

``GMP::gmp``
  The GMP shared library, carrying the ``gmp.h`` directory as a usage
  requirement when the headers were found.

Result variables
^^^^^^^^^^^^^^^^

``GMP_FOUND``, ``GMP_LIBRARY``, ``GMP_INCLUDE_DIR``, ``GMP_VERSION``.
#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_path(GMP_INCLUDE_DIR
  NAMES gmp.h
  DOC "Directory containing gmp.h")

block()
  # Never accept libgmp.a: only the dynamically linked library may be
  # redistributed under the terms CUDA-Q relies on.
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
  find_library(GMP_LIBRARY
    NAMES gmp
    HINTS "${CUDAQ_LIBRARY_DIR}"
    DOC "GMP shared library")
endblock()

unset(GMP_VERSION)
if(EXISTS "${GMP_INCLUDE_DIR}/gmp.h")
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
  set(GMP_Headers_FOUND TRUE)
endif()

find_package_handle_standard_args(GMP
  REQUIRED_VARS GMP_LIBRARY
  VERSION_VAR GMP_VERSION
  HANDLE_COMPONENTS
  REASON_FAILURE_MESSAGE
  "Run scripts/install_prerequisites.sh to build GMP from source, or install \
it with your package manager (libgmp-dev on apt, gmp on brew) and set GMP_ROOT \
or the GMP_INSTALL_PREFIX environment variable.")

if(GMP_FOUND)
  if(NOT TARGET GMP::gmp)
    add_library(GMP::gmp SHARED IMPORTED)
    set_property(TARGET GMP::gmp PROPERTY IMPORTED_LOCATION "${GMP_LIBRARY}")
  endif()
  if(GMP_INCLUDE_DIR)
    set_property(TARGET GMP::gmp PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}")
  endif()
endif()
