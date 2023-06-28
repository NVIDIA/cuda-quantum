# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-chemistry SHARED molecule.cpp)
target_link_libraries(cudaq-chemistry PRIVATE cudaq-spin)
target_include_directories(cudaq-chemistry SYSTEM
   PRIVATE 
     ${CMAKE_SOURCE_DIR}/tpls/xtl/include 
     ${CMAKE_SOURCE_DIR}/tpls/xtensor/include)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(cudaq-chemistry PRIVATE --param=evrp-mode=legacy)
endif()

install(TARGETS cudaq-chemistry DESTINATION lib)