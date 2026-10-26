/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_DELAYMEASUREMENTS
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "delay-measurements"

using namespace mlir;

namespace {

/// Delay Measurements
///
/// This pass delays measurements as long as possible. This reordering of
/// instructions is useful for Base Profile programs where the user measures 1
/// qubit early in the program and then performs quantum operations on other
/// qubits later in the program.
struct DelayMeasurementsPass
    : public cudaq::opt::impl::DelayMeasurementsBase<DelayMeasurementsPass> {
  using DelayMeasurementsBase::DelayMeasurementsBase;

  void runOnOperation() override {

    func::FuncOp func = getOperation();
    auto &blocks = func.getBlocks();

    if (blocks.empty())
      return;

    // If the function doesn't have measurements, we can ignore it.
    if (!func.walk([](Operation *op) {
               if (op->hasTrait<cudaq::QuantumMeasure>())
                 return WalkResult::interrupt();
               return WalkResult::advance();
             })
             .wasInterrupted())
      return;

    if (!func.getFunctionBody().hasOneBlock()) {
      func.emitError("DelayMeasurementsPass cannot handle multiple blocks. Do "
                     "you have if statements in a Base Profile QIR program?");
      signalPassFailure();
      return;
    }

    moveMeasurementsToEnd(*blocks.begin());
  }

  /// Collect the reverse of the original recursive preorder, keeping only
  /// each operation's last occurrence. Moving an operation repeatedly only
  /// leaves its last move visible, so this preserves the existing order.
  void addOpAndUsersToList(Operation *op,
                           SmallVectorImpl<Operation *> &opsToMoveToEnd,
                           DenseSet<Operation *> &visited) {
    SmallVector<std::pair<Operation *, bool>> worklist{{op, false}};
    while (!worklist.empty()) {
      auto [current, expanded] = worklist.pop_back_val();
      if (expanded) {
        opsToMoveToEnd.push_back(current);
        continue;
      }
      if (!visited.insert(current).second)
        continue;
      worklist.emplace_back(current, true);
      // The stack visits users in reverse order, before their definition.
      for (auto *user : current->getUsers())
        worklist.emplace_back(user, false);
    }
  }

  /// The Base Profile requires that irreversible operations (i.e.
  /// measurements) come after reversible operations. This function enforces
  /// that.
  /// @param mainBlock block to process
  void moveMeasurementsToEnd(Block &mainBlock) {
    SmallVector<Operation *> roots;

    // Keep track of which qubits have been measured as we're walking through
    // the block
    DenseSet<Value> measuredQubits;

    // Step 1: Identify roots of the use-def subgraphs to move.
    for (auto &op : mainBlock) {
      if (op.hasTrait<cudaq::QuantumMeasure>()) {
        // Save the fact that we're measuring these qubits
        for (auto operand : op.getOperands())
          measuredQubits.insert(operand);
      }

      if (op.hasTrait<cudaq::QuantumMeasure>() || isa<func::ReturnOp>(op) ||
          isa<cudaq::quake::DeallocOp>(op)) {
        roots.push_back(&op);
        continue;
      }

      // Check to see if this operation has arguments that were already
      // measured. If so, add this operation to opsToMoveToEnd, too.
      for (auto operand : op.getOperands()) {
        if (measuredQubits.find(operand) != measuredQubits.end()) {
          roots.push_back(&op);
          break;
        }
      }
    }

    // Step 2: Visit each operation once, starting with the last root, then
    // reverse the collected order to reproduce the original sequence of moves.
    SmallVector<Operation *> opsToMoveToEnd;
    DenseSet<Operation *> visited;
    for (auto *root : llvm::reverse(roots))
      addOpAndUsersToList(root, opsToMoveToEnd, visited);
    for (Operation *opToMove : llvm::reverse(opsToMoveToEnd))
      mainBlock.getOperations().splice(
          mainBlock.end(), mainBlock.getOperations(), opToMove->getIterator());
  }
};
} // namespace
