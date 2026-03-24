/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace cudaq {

// Add all required dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    // MLIR dialects
    mlir::arith::ArithDialect,
    mlir::cf::ControlFlowDialect,
    mlir::complex::ComplexDialect,
    mlir::func::FuncDialect,
    mlir::LLVM::LLVMDialect,
    mlir::math::MathDialect,

    // CUDA-Q dialects
    cudaq::cc::CCDialect,
    quake::QuakeDialect
  >();
  // clang-format on
}

} // namespace cudaq
