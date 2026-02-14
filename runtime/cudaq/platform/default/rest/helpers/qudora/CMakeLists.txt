# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE QudoraServerHelper.cpp)
add_target_config(qudora)

add_library(cudaq-serverhelper-qudora SHARED QudoraServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-qudora
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
  PRIVATE
    LLVMSupport
)
install(TARGETS cudaq-serverhelper-qudora DESTINATION lib)

