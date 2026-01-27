# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(SERVICE_COMPONENTS braket s3-crt sts)
find_package(AWSSDK REQUIRED COMPONENTS ${SERVICE_COMPONENTS})
target_sources(cudaq-rest-qpu PRIVATE BraketExecutor.cpp BraketServerHelper.cpp)
add_target_config(braket)
add_library(cudaq-serverhelper-braket SHARED BraketExecutor.cpp BraketServerHelper.cpp)

target_link_libraries(cudaq-serverhelper-braket
    cudaq-common
    fmt::fmt-header-only
    ${AWSSDK_LINK_LIBRARIES})

install(TARGETS cudaq-serverhelper-braket DESTINATION lib)
