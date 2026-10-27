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

# Define the dependencies of `trivial-mlir`. In a shared-provider SDK
# (installed MLIRConfig defines the shared aggregate targets) every
# MLIR/LLVM symbol resolves from the provider DSOs — linking the upstream
# component archives here would embed a second full MLIR implementation
# into the extension, hidden by --exclude-libs but rejected by the SDK's
# pre-strip audit. Only the legacy static SDK uses the component archives.
if(TARGET MLIR)
  target_link_libraries(trivial-mlir PUBLIC
    MLIR MLIRExecutionEngineShared LLVM)
else()
  target_link_libraries(trivial-mlir PUBLIC
    MLIRIR MLIRPass MLIRSupport MLIRFuncDialect MLIRTransformUtils)
endif()

# Make sure this static library can be embedded into the shared Python extension.
set_target_properties(trivial-mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
