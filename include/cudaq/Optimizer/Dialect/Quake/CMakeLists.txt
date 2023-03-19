# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_dialect(Quake quake)
add_cudaq_interface(QuakeInterfaces)
add_cudaq_dialect_doc(QuakeDialect quake)

set(LLVM_TARGET_DEFINITIONS Canonical.td)
mlir_tablegen(Canonical.inc -gen-rewriters)
add_public_tablegen_target(CanonicalIncGen)
