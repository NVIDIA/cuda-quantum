# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-fermioniq-qpu PRIVATE FermioniqServerHelper.cpp)

add_library(cudaq-serverhelper-fermioniq SHARED FermioniqServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-fermioniq
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-fermioniq DESTINATION lib)

