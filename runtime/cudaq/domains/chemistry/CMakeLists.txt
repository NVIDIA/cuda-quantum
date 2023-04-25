# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-chemistry SHARED molecule.cpp)
target_link_libraries(cudaq-chemistry PRIVATE cudaq-spin)
target_include_directories(cudaq-chemistry 
   PRIVATE 
     ${CMAKE_SOURCE_DIR}/tpls/xtl/include 
     ${CMAKE_SOURCE_DIR}/tpls/xtensor/include)

install(TARGETS cudaq-chemistry DESTINATION lib)