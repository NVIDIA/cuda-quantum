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

/// A `cc.scope` result that carries a chain wire out of the scope, and the
/// `cc.continue` operand that feeds it.
struct SingleQubitChainScopeStep {
  mlir::Value wire;
  mlir::OpOperand *continueOperand;
};

/// A maximal sequence of gates on one target qubit, in execution order, with
/// the same controls throughout. Each operation threads the `!quake.wire`
/// values of the previous one. A chain may cross single-block `cc.scope`
/// operations, recorded per lane in `scopeSteps`.
struct SingleQubitChain {
  llvm::SmallVector<mlir::Operation *> operations;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
  llvm::SmallVector<llvm::SmallVector<SingleQubitChainScopeStep>> scopeSteps;
};

/// Returns whether an operation may be part of a chain.
using SingleQubitChainPredicate = llvm::function_ref<bool(mlir::Operation *)>;
/// Returns whether `next` may extend the chain that starts at `head`.
using SingleQubitChainContinuation =
    llvm::function_ref<bool(mlir::Operation *head, mlir::Operation *next)>;

/// Collect the maximal chains of operations accepted by `isSupported` without
/// mutating the IR. `canContinue` receives the chain head and the next
/// operation and may end a chain early, for example when the two operations
/// have different control predicates. The returned chains borrow IR pointers
/// and values, so the caller must not mutate the block before rewriting the
/// chains it selects in reverse collection order.
llvm::SmallVector<SingleQubitChain, 0>
collectSingleQubitChains(mlir::Block &block,
                         SingleQubitChainPredicate isSupported,
                         SingleQubitChainContinuation canContinue);

/// Replace the outputs of `chain` with `replacements`, which must match
/// `chain.outputs` in count, type, and dominance, and erase the chain's
/// operations.
void replaceSingleQubitChain(SingleQubitChain &&chain,
                             mlir::ValueRange replacements);

} // namespace cudaq::opt
