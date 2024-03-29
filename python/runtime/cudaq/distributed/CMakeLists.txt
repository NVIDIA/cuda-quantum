# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

message(STATUS "Building Python MPI Comm plugin based on mpi4py")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/plugins)
set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
# IMPORTANT: Don't change this lib name without updating the getMpiPlugin function
set(LIBRARY_NAME cudaq-py-comm-plugin)
add_library(${LIBRARY_NAME} SHARED mpi_comm_impl.cpp)
if (SKBUILD)
    target_link_libraries(${LIBRARY_NAME} PRIVATE pybind11::pybind11 Python::Module)
    target_link_options(${LIBRARY_NAME} PRIVATE -Wl,--unresolved-symbols=ignore-in-object-files)
else()
    if (NOT Python_FOUND)
      message(FATAL_ERROR "find_package(Python) not run?")
    endif()
    target_link_libraries(${LIBRARY_NAME} PRIVATE pybind11::pybind11 Python::Python)
endif()
target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/runtime/cudaq/distributed)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib/plugins)
