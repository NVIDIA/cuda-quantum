# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(IR)
add_subdirectory(Transforms)

add_custom_target(MLIRQLXIncGen DEPENDS
  MLIRQLXDialectIncGen
  MLIRQLXTypesIncGen
  MLIRQLXAttrsIncGen
  MLIRQLXEnumsIncGen
  MLIRQLXOpsIncGen
)
