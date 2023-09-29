# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/plugins)
SET(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")

add_library(cudaq-pyscf SHARED PySCFDriver.cpp)
target_link_libraries(cudaq-pyscf PRIVATE pybind11::embed cudaq-chemistry cudaq-spin cudaq cudaq-py-utils)
install(TARGETS cudaq-pyscf DESTINATION lib/plugins)