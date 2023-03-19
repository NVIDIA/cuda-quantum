# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_dialect_library(QTXDialect
  QTXDialect.cpp
  QTXInterfaces.cpp
  QTXOps.cpp
  QTXStructure.cpp
  QTXTypes.cpp

  DEPENDS
  QTXDialectIncGen
  QTXOpsIncGen
  QTXTypesIncGen

  LINK_LIBS
  DialectCommon
  MLIRFuncDialect
  MLIRIR
)

