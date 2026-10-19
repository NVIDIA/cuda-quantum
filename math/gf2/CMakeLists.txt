# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-gf2 lib/gf2.c)
add_library(cudaq::gf2 ALIAS cudaq-gf2)

target_sources(cudaq-gf2 PUBLIC
  FILE_SET HEADERS
  BASE_DIRS include
  FILES include/cudaq/gf2/core.h
)
target_compile_features(cudaq-gf2 PUBLIC c_std_11)
set_target_properties(cudaq-gf2 PROPERTIES
  EXPORT_NAME gf2
  C_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)
