# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
target_sources(cudaq-rest-qpu PRIVATE IQMServerHelper.cpp)
add_target_config(iqm)
add_target_mapping_arch(iqm Adonis.txt)
add_target_mapping_arch(iqm Apollo.txt)
