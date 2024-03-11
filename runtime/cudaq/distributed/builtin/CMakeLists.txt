# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if (MPI_CXX_FOUND)
    message(STATUS "Building default MPI Comm plugin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/plugins)
    set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
    # IMPORTANT: Don't change this lib name without updating the getMpiPlugin function
    set(LIBRARY_NAME cudaq-comm-plugin)
    add_library(${LIBRARY_NAME} SHARED mpi_comm_impl.cpp)
    target_link_libraries(${LIBRARY_NAME} PRIVATE MPI::MPI_CXX)
    target_include_directories(${LIBRARY_NAME} PRIVATE ..)
    install(TARGETS ${LIBRARY_NAME} DESTINATION lib/plugins)
endif()