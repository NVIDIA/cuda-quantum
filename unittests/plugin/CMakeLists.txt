# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_llvm_pass_plugin(CustomPassPlugin CustomPassPlugin.cpp)
# Depends on QuakeDialect TableGen to use the generated `.h.inc` files.
add_dependencies(CustomPassPlugin QuakeDialect)
