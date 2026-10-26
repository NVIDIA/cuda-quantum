# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_public_c_api_library(QLXCAPIDialectQLX
  QLX.cpp
  DISABLE_INSTALL

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRQLXDialect
  MLIRQLXTransforms
)

add_mlir_public_c_api_library(QLXCAPIDialectLVM
  LVM.cpp
  DISABLE_INSTALL
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRCAPIIR MLIRLVMDialect)

add_mlir_public_c_api_library(QLXCAPIDialectCflow
  Cflow.cpp
  DISABLE_INSTALL
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRCAPIIR MLIRCflowDialect)

add_mlir_public_c_api_library(QLXCAPIDialectEvent
  Event.cpp
  DISABLE_INSTALL
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRCAPIIR MLIREventDialect)

add_mlir_public_c_api_library(QLXCAPIDialectPhys
  Phys.cpp
  DISABLE_INSTALL
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRCAPIIR MLIRPhysDialect)

add_mlir_public_c_api_library(QLXCAPIDialectFabric
  Fabric.cpp
  DISABLE_INSTALL

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRFabricDialect
)
