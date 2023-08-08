# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")

add_library(cudaq-py-utils SHARED LinkedLibraryHolder.cpp)
target_link_libraries(cudaq-py-utils PRIVATE nvqir cudaq fmt::fmt-header-only)
target_include_directories(cudaq-py-utils PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/python/utils>)
install(TARGETS cudaq-py-utils DESTINATION lib)