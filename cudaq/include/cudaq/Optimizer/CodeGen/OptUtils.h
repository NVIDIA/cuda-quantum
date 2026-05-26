//===- OptUtils.h - MLIR Execution Engine opt pass utilities ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Additional modifications by NVIDIA Corporation.
// - Use cudaq namespace instead of mlir namespace.
// - Add an allowVectorization parameter to makeOptimizingTransformer.
//
//===----------------------------------------------------------------------===//
//
// This file declares the utility functions to trigger LLVM optimizations from
// MLIR Execution Engine.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <functional>

namespace llvm {
class Module;
class Error;
class TargetMachine;
} // namespace llvm

namespace cudaq {

/// Create a module transformer function for MLIR ExecutionEngine that runs
/// LLVM IR passes corresponding to the given speed and size optimization
/// levels (e.g. -O2 or -Os). If not null, `targetMachine` is used to
/// initialize passes that provide target-specific information to the LLVM
/// optimizer. `targetMachine` must outlive the returned std::function.
/// Note: this is a modified version of mlir::makeOptimizingTransformer that
/// disables vectorization by default.
std::function<llvm::Error(llvm::Module *)>
makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel,
                          llvm::TargetMachine *targetMachine,
                          bool allowVectorization = false);
} // namespace cudaq
