# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(include/qlx/Conversion)
add_subdirectory(include/qlx/Dialect/Fabric)
add_subdirectory(include/qlx/Dialect/QLX)
add_subdirectory(include/qlx/Dialect/LVM)

# Per-dialect implementation (IR + Transforms).
add_subdirectory(lib/Dialect/Fabric)
add_subdirectory(lib/Dialect/QLX)
add_subdirectory(lib/Dialect/LVM)

# Cross-dialect lowerings (MLIR-style Conversion/<X>To<Y>).
add_subdirectory(lib/Conversion/QuakeToQLX)
add_subdirectory(lib/Conversion/QLXToLVM)

# Translation backends (MLIR-style Target/).
add_subdirectory(lib/Target/Fabric)

# C-API shims for the Python bindings (always built; pure additive layer).
add_subdirectory(lib/CAPI)

# Tools.
add_subdirectory(tools/qlx-opt)
add_subdirectory(tools/qlx-translate)
