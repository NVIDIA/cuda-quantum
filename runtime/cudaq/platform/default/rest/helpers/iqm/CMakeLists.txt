# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE IQMServerHelper.cpp)
add_target_config(iqm)
add_target_mapping_arch(iqm Adonis.txt)
add_target_mapping_arch(iqm Apollo.txt)
add_target_mapping_arch(iqm Aphrodite.txt)


add_library(cudaq-serverhelper-iqm SHARED IQMServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-iqm
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-iqm DESTINATION lib)

