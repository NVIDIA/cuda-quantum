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

# Compile with the same flags as LLVM to ensure ABI compatibility.
llvm_update_compile_flags(trivial-mlir)

# Define the dependencies of `trivial-mlir` using the usual static MLIR components.
# When `trivial-mlir` is pulled into the shared Python extension, the MLIR/LLVM symbols
# will be resolved from the single cudaq::MLIR DSO instead.
target_link_libraries(trivial-mlir PUBLIC
  MLIRIR MLIRPass MLIRSupport MLIRFuncDialect MLIRTransformUtils)

# Make sure this static library can be embedded into the shared Python extension.
set_target_properties(trivial-mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
