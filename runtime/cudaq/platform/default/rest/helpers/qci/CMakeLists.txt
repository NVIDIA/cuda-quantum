# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors.  #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

target_sources(cudaq-rest-qpu PRIVATE QCIServerHelper.cpp)
add_target_config(qci)

add_library(cudaq-serverhelper-qci SHARED QCIServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-qci
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-qci DESTINATION lib)

