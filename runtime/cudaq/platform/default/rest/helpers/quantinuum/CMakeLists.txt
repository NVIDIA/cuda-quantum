# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

target_sources(cudaq-rest-qpu PRIVATE QuantinuumServerHelper.cpp)
add_target_config(quantinuum)

add_library(cudaq-serverhelper-quantinuum SHARED QuantinuumServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-quantinuum
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-quantinuum DESTINATION lib)

## Following should be deleted once Nexus-based API is fully implemented
target_sources(cudaq-rest-qpu PRIVATE QuantinuumLegacyServerHelper.cpp)
add_target_config(quantinuum_legacy)
add_library(cudaq-serverhelper-quantinuum_legacy SHARED QuantinuumLegacyServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-quantinuum_legacy
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-quantinuum_legacy DESTINATION lib)
