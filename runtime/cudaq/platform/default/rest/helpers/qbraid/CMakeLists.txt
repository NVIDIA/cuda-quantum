# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE QbraidServerHelper.cpp)
add_target_config(qbraid)

add_library(cudaq-serverhelper-qbraid SHARED QbraidServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-qbraid
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-qbraid DESTINATION lib)