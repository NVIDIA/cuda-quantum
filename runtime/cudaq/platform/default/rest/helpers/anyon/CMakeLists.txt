# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE AnyonServerHelper.cpp)
add_target_config(anyon)
add_target_mapping_arch(anyon berkeley-25q.txt)
add_target_mapping_arch(anyon telegraph-8q.txt)

add_library(cudaq-serverhelper-anyon SHARED AnyonServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-anyon
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-anyon DESTINATION lib)

