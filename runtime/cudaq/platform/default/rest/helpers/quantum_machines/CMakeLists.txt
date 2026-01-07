# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE QuantumMachinesServerHelper.cpp)
add_target_config(quantum_machines)

add_library(cudaq-serverhelper-quantum_machines SHARED QuantumMachinesServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-quantum_machines
  PUBLIC
    cudaq-common
    fmt::fmt-header-only
)
install(TARGETS cudaq-serverhelper-quantum_machines DESTINATION lib)
