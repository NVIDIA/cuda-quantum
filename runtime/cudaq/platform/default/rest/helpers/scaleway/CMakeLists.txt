# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE ScalewayServerHelper.cpp)
add_target_config(scaleway)


add_library(cudaq-serverhelper-scaleway SHARED IQMServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-scaleway
  PUBLIC
    cudaq-common
    fmt::fmt-header-only
)
install(TARGETS cudaq-serverhelper-scaleway DESTINATION lib)

