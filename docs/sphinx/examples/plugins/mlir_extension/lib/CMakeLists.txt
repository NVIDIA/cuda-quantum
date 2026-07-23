# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The MLIR extension library.
add_library(trivial-mlir STATIC
  TrivialDialect.cpp
  TrivialOps.cpp
  TrivialPasses.cpp)

# Generated headers land under ${PROJECT_BINARY_DIR}/include/Trivial; the source
# headers live under ../include. Both are needed to compile these sources.
target_include_directories(trivial-mlir PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${PROJECT_BINARY_DIR}/include)

add_dependencies(trivial-mlir
  TrivialDialectIncGen
  TrivialOpsIncGen
  TrivialPassesIncGen)

# Resolve MLIR/LLVM dynamically from the single libcudaqMLIR instance.
target_link_libraries(trivial-mlir PUBLIC cudaq::MLIR)

# Make sure this static library can be embedded into the shared Python extension.
set_target_properties(trivial-mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
