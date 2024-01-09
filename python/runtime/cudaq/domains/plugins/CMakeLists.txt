# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

message(STATUS "Building cudaq-pyscf plugin.")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/plugins)
SET(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
add_library(cudaq-pyscf SHARED PySCFDriver.cpp)
if (SKBUILD)
    target_link_libraries(cudaq-pyscf PRIVATE pybind11::pybind11 Python::Module cudaq-chemistry cudaq-spin cudaq cudaq-py-utils)
    target_link_options(cudaq-pyscf PRIVATE -Wl,--unresolved-symbols=ignore-in-object-files)
else()
    find_package(Python 3 COMPONENTS Development.Embed REQUIRED)
    target_link_libraries(cudaq-pyscf PRIVATE Python::Python pybind11::pybind11 cudaq-chemistry cudaq-spin cudaq cudaq-py-utils)
endif()
install(TARGETS cudaq-pyscf DESTINATION lib/plugins)