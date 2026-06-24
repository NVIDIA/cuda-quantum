# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Minizip is an addon library bundled with zlib (contrib/minizip) but not
# installed by default.  This project's install_prerequisites.sh builds it.
# The module tries pkg-config first, then falls back to manual search.

set(_minizip_hints
  "${ZLIB_ROOT}"
  "$ENV{ZLIB_INSTALL_PREFIX}"
  "${ZLIB_INCLUDE_DIR}/.."
)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  find_path(_minizip_pc_dir NAMES minizip.pc
    HINTS ${_minizip_hints}
    PATH_SUFFIXES lib/pkgconfig
  )
  if(_minizip_pc_dir)
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${_minizip_pc_dir}")
  endif()
  pkg_check_modules(_minizip_pkg QUIET minizip)
endif()

find_library(Minizip_LIBRARY NAMES minizip
  HINTS
    ${_minizip_pkg_LIBRARY_DIRS}
    ${_minizip_hints}
  PATH_SUFFIXES lib
)

find_path(Minizip_INCLUDE_DIR NAMES unzip.h
  HINTS
    ${_minizip_pkg_INCLUDE_DIRS}
    ${_minizip_hints}
  PATH_SUFFIXES include include/minizip
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Minizip
  REQUIRED_VARS Minizip_LIBRARY Minizip_INCLUDE_DIR
)

if(Minizip_FOUND AND NOT TARGET Minizip::Minizip)
  add_library(Minizip::Minizip UNKNOWN IMPORTED)
  set_target_properties(Minizip::Minizip PROPERTIES
    IMPORTED_LOCATION "${Minizip_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Minizip_INCLUDE_DIR}"
  )
  # Minizip depends on zlib
  find_package(ZLIB QUIET)
  if(TARGET ZLIB::ZLIB)
    set_property(TARGET Minizip::Minizip APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES ZLIB::ZLIB
    )
  endif()
endif()
