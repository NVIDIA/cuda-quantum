# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/oqc.config
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/oqc.config)

target_sources(cudaq-rest-qpu PRIVATE OQCServerHelp.cpp)
add_target_config(oqc)

