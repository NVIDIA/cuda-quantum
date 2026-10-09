# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

qlx_select_mlir_link_libraries(_qlx_mlir_link_libraries
  MLIRIR
  MLIRSupport
)

add_mlir_library(MLIRCUDAQLogicalInterfaces
  SemanticInterfaces.cpp
  PhysicalInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/CUDAQLogical/Interfaces

  DEPENDS
  MLIRCUDAQLogicalInterfacesIncGen

  LINK_LIBS PUBLIC
  ${_qlx_mlir_link_libraries}
)
