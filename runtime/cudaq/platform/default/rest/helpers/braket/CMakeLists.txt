# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(SERVICE_COMPONENTS braket s3-crt sts)
find_package(AWSSDK REQUIRED COMPONENTS ${SERVICE_COMPONENTS})
# Only this plugin builds these sources. The AWS SDK is linked statically, so
# compiling them into `cudaq-rest-qpu` as well would put a second copy of the
# CRT allocator state into the process, and selecting `braket` then `quera`
# would abort in aws-c-common on `allocator != NULL`.
add_target_config(braket)
add_library(cudaq-serverhelper-braket SHARED BraketExecutor.cpp BraketServerHelper.cpp)

target_link_libraries(cudaq-serverhelper-braket
    cudaq-common
    cudaq-logger
    fmt::fmt-header-only
    ${AWSSDK_LINK_LIBRARIES})

install(TARGETS cudaq-serverhelper-braket DESTINATION lib
        COMPONENT Runtime)
