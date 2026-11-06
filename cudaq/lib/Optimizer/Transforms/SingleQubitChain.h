/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace cudaq::opt {

/// A maximal sequence of gates on one target qubit, in execution order, with
/// the same controls and control polarities throughout. Each operation threads
/// the `!quake.wire` values of the previous one, possibly through the results
/// of single-block `cc.scope` operations.
struct SingleQubitChain {
  llvm::SmallVector<mlir::Operation *> operations;
  llvm::SmallVector<mlir::Value> inputs;
};

/// Returns whether an operation may be part of a chain.
using SingleQubitChainPredicate = llvm::function_ref<bool(mlir::Operation *)>;

/// Collect the maximal chains of operations accepted by `isSupported` without
/// mutating the IR. A chain continues only while the next gate uses the same
/// control wires in the same positions with the same polarities. The returned
/// chains borrow IR pointers and values, so the caller must not mutate the
/// block before rewriting the chains it selects in reverse collection order.
llvm::SmallVector<SingleQubitChain, 0>
collectSingleQubitChains(mlir::Block &block,
                         SingleQubitChainPredicate isSupported);

/// Replace the chain with `replacements`, which take the place of the wire
/// results of its first operation and must dominate every use of the chain's
/// results, and erase the chain's operations.
void replaceSingleQubitChain(SingleQubitChain &&chain,
                             mlir::ValueRange replacements);

/// Call `rewriteBlock` on each block of `region`, then visit the regions of
/// the operations that remain in that block. A rewrite in a nested block may
/// erase operations that follow the enclosing operation, when a chain leaves a
/// `cc.scope`, so the next operation is read only after the nested visit
/// returns. `Operation::walk` reads it in advance and is not safe here.
void forEachNestedBlock(mlir::Region &region,
                        llvm::function_ref<void(mlir::Block &)> rewriteBlock);

} // namespace cudaq::opt
