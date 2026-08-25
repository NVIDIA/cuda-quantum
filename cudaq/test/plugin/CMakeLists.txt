# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)
add_llvm_pass_plugin(CustomPassPlugin CustomPassPlugin.cpp)
# Depends on QuakeDialect TableGen to use the generated `.h.inc` files.
add_dependencies(CustomPassPlugin QuakeDialect)
# Link no CUDA-Q or MLIR library: cudaq-opt links MLIR/LLVM statically and
# re-exports it for plugins, so every symbol resolves from the host. Linking
# libcudaqMLIR would pull a second copy of LLVM's globals into the process and
# abort on duplicate cl::opt registration.
