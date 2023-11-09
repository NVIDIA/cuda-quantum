# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_public_c_api_library(CUDAQuantumMLIRCAPI 
  Dialects.cpp

  LINK_LIBS PRIVATE 
  QuakeDialect 
  CCDialect
)