/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Common/Traits.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/CC/CCOps.h.inc"

namespace cudaq::cc {

template <typename A>
using ComputePtrIndicesAdaptor = mlir::LLVM::GEPIndicesAdaptor<A>;

class RegionBuilderGuard : mlir::OpBuilder::InsertionGuard {
  using Base = mlir::OpBuilder::InsertionGuard;

public:
  /// Helper for callbacks when building a Region in a control-flow Op. This
  /// sets up the entry block, block arguments, and sets the insertion point of
  /// \p builder to the start of the new block. The calling context can get a
  /// poiner to the new block by calling `builder.getBlock()`. Since this
  /// extends InsertionGuard, an instance will preserve the insertion point of
  /// \p builder as a bonus.
  explicit RegionBuilderGuard(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Region &region, mlir::TypeRange blockArgTys)
      : Base(builder) {
    region.push_back(new mlir::Block());
    auto &block = region.front();
    for (auto ty : blockArgTys)
      block.addArgument(ty, loc);
    builder.setInsertionPointToStart(&block);
  }
};
} // namespace cudaq::cc
