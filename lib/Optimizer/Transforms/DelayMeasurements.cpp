/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

#define DEBUG_TYPE "delay-measurements"

using namespace mlir;

namespace cudaq::opt {

/// Delay Measurements
///
/// This pass delays measurements as long as possible. This reordering of
/// instructions is useful for Base Profile programs where the user measures 1
/// qubit early in the program and then performs quantum operations on other
/// qubits later in the program.
struct DelayMeasurementsPass
    : public cudaq::opt::DelayMeasurementsBase<DelayMeasurementsPass> {
  // explicit DelayMeasurementsPass() = default;

  void runOnOperation() override {

    auto func = getOperation();
    auto &blocks = func.getBlocks();

    if (blocks.size() == 0)
      return;

    if (blocks.size() > 1) {
      func.emitError("DelayMeasurementsPass cannot handle multiple blocks. Do "
                     "you have if statements in a Base Profile QIR program?");
      signalPassFailure();
      return;
    }

    moveMeasurementsToEnd(*blocks.begin());
  }

  /// @brief Add `op` and all of its users into `opsToMoveToEnd`
  void
  addOpAndUsersToList(Operation &op,
                      llvm::SmallVector<mlir::Operation *> &opsToMoveToEnd) {
    opsToMoveToEnd.push_back(&op);
    for (auto user : op.getUsers())
      addOpAndUsersToList(*user, opsToMoveToEnd);
  }

  /// @brief The Base Profile requires that irreversible operations (i.e.
  /// measurements) come after reversible operations. This function enforces
  /// that.
  /// @param mainBlock block to process
  void moveMeasurementsToEnd(mlir::Block &mainBlock) {
    llvm::SmallVector<mlir::Operation *> opsToMoveToEnd;

    // Keep track of which qubits have been measured as we're walking through
    // the block
    llvm::DenseSet<mlir::Value> measuredQubits;

    // Step 1: Identify operations to move. Add to opsToMoveToEnd.
    for (auto &op : mainBlock) {
      if (op.hasTrait<QuantumMeasure>()) {
        // Save the fact that we're measuring these qubits
        for (auto operand : op.getOperands())
          measuredQubits.insert(operand);
      }

      if (op.hasTrait<QuantumMeasure>() || isa<mlir::func::ReturnOp>(op) ||
          isa<quake::DeallocOp>(op)) {
        addOpAndUsersToList(op, opsToMoveToEnd);
        continue;
      }

      // Check to see if this operation has arguments that were already
      // measured. If so, add this operation to opsToMoveToEnd, too.
      for (auto operand : op.getOperands()) {
        if (measuredQubits.find(operand) != measuredQubits.end()) {
          addOpAndUsersToList(op, opsToMoveToEnd);
          break;
        }
      }
    }

    // Step 2: Sequentially move identified operations to the end of the block
    for (mlir::Operation *opToMove : opsToMoveToEnd)
      mainBlock.getOperations().splice(
          mainBlock.end(), mainBlock.getOperations(), opToMove->getIterator());
  }
};
} // namespace cudaq::opt

std::unique_ptr<Pass> cudaq::opt::createDelayMeasurementsPass() {
  return std::make_unique<DelayMeasurementsPass>();
}
