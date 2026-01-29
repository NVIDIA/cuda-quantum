# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_CXX_COMPILER "${CMAKE_BINARY_DIR}/bin/nvq++")

set(CMAKE_CXX_COMPILE_OBJECT
  "<CMAKE_CXX_COMPILER> -fPIC --enable-mlir --disable-mlir-links <DEFINES> <INCLUDES> -o <OBJECT> -c <SOURCE>")

# FIXME Error with SHARED, it pulls in all the mlir libraries anyway
add_library(quantum_lib
  OBJECT
    quantum_lib.cpp
)

target_include_directories(quantum_lib
  PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

add_dependencies(quantum_lib
  nvq++ cudaq-opt cudaq-quake cudaq-translate
)
