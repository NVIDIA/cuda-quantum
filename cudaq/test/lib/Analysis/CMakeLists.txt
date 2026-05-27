# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_library(CUDAQTestAnalysis
  TestUnitaryOpGrouping.cpp

  DISABLE_INSTALL

  DEPENDS
    QuakeDialect
    OptAnalysis

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    OptAnalysis
)

if(TARGET cudaq-opt)
  target_link_libraries(cudaq-opt PRIVATE CUDAQTestAnalysis)
  target_compile_definitions(cudaq-opt PRIVATE CUDAQ_INCLUDE_TESTS)
endif()
