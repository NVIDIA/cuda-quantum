# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)

add_llvm_executable(CircuitCheck
  CircuitCheck.cpp
)

target_include_directories(CircuitCheck
  PRIVATE ${CMAKE_SOURCE_DIR}/tpls/eigen
  PRIVATE ${CMAKE_SOURCE_DIR}/runtime
)

# The UnitaryBuilder engine now lives in the OptAnalysis library. CircuitCheck is
# one CLI client of it.
target_link_libraries(CircuitCheck
  PRIVATE
  MLIRArithDialect
  MLIRParser
  MLIRPass
  MLIRTransforms

  CCDialect
  QuakeDialect
  OptAnalysis
)
