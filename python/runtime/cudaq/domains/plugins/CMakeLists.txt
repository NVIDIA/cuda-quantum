# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

message(STATUS "Building cudaq-pyscf plugin.")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/plugins)
if(APPLE)
  SET(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/..")
else()
  SET(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
endif()
add_library(cudaq-pyscf SHARED PySCFDriver.cpp)

# Use Python::Module (not Python::Python) so we don't bake an absolute path to
# libpython into the shared library. Python symbols resolve at runtime from
# whatever interpreter loads the module. This makes the plugin relocatable on
# both macOS (no framework path) and Linux (no direct libpython link).
target_link_libraries(cudaq-pyscf
  PRIVATE
    pybind11::pybind11 Python::Module
    cudaq-chemistry cudaq-operator cudaq cudaq-py-utils cudaq-platform-default)
# Apple's ld64 handles undefined symbols via -undefined dynamic_lookup
# (set automatically by Python::Module). GNU ld needs an explicit flag.
if (NOT APPLE)
  target_link_options(cudaq-pyscf
    PRIVATE -Wl,--unresolved-symbols=ignore-in-object-files)
endif()

install(TARGETS cudaq-pyscf DESTINATION lib/plugins)
