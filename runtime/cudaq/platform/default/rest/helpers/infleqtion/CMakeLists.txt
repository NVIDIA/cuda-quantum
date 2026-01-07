# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE InfleqtionServerHelper.cpp)
add_target_config(infleqtion)

add_library(cudaq-serverhelper-infleqtion SHARED InfleqtionServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-infleqtion
  PUBLIC
    cudaq-common
    fmt::fmt-header-only
)
install(TARGETS cudaq-serverhelper-infleqtion DESTINATION lib)
