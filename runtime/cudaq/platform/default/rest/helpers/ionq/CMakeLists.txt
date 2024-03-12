# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE IonQServerHelper.cpp)
add_target_config(ionq)

add_library(cudaq-serverhelper-ionq SHARED IonQServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-ionq
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-ionq DESTINATION lib)

