# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

target_sources(cudaq-rest-qpu PRIVATE LQCloudServerHelper.cpp)

add_library(cudaq-serverhelper-lqcloud SHARED LQCloudServerHelper.cpp)
target_link_libraries(cudaq-serverhelper-lqcloud
  PUBLIC
    cudaq-common
    cudaq-logger
)
install(TARGETS cudaq-serverhelper-lqcloud DESTINATION lib COMPONENT Runtime)
