/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ADDMEASUREMENTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "add-measurements"

using namespace mlir;

namespace {

/// Analysis class that examines a function to determine whether it contains
/// measurement operations and collects all qubit allocations. Also, gather all
/// the returns for redirection
struct Analysis {
  Analysis() = default;

  explicit Analysis(func::FuncOp func) {
    func.walk([&](Operation *op) {
      if (op->hasTrait<cudaq::QuantumMeasure>()) {
        hasMeasurement = true;
        return WalkResult::interrupt();
      }
      if (isa<quake::AllocaOp>(op))
        allocations.emplace_back(op);
      else if (isa<func::ReturnOp>(op))
        returns.emplace_back(op);
      return WalkResult::advance();
    });
  }

  bool hasMeasurement = false;
  SmallVector<quake::AllocaOp> allocations;
  SmallVector<func::ReturnOp> returns;

  bool hasQubitAlloca() const { return !allocations.empty(); }
};

/// Add measurement operations for all allocated qubits in a function.
/// This transformation creates a new block at the end of the function,
/// redirects all return operations to branch to this block, adds `quake.mz`
/// measurement operations for each qubit allocation, and adds a final return.
/// For vector allocations, the measurements are collected into a vector of
/// measurement results.
LogicalResult
addMeasurements(func::FuncOp funcOp, SmallVector<quake::AllocaOp> &allocations,
                const SmallVector<func::ReturnOp> &returnsToReplace) {
  auto loc = funcOp.getLoc();
  auto ctx = funcOp.getContext();
  OpBuilder builder(ctx);

  // Create a new block at the end of the function.
  Block *newBlock = funcOp.addBlock();

  // Add block arguments for return values if the function returns anything
  ArrayRef<Type> returnTypes = funcOp.getFunctionType().getResults();
  if (!returnTypes.empty()) {
    SmallVector<Location> argLocs(returnTypes.size(), loc);
    newBlock->addArguments(returnTypes, argLocs);
  }

  // Replace every func.return in the function with a branch to the new block.
  for (auto returnOp : returnsToReplace) {
    OpBuilder builder(returnOp);
    builder.create<cf::BranchOp>(returnOp.getLoc(), newBlock,
                                 returnOp.getOperands());
    returnOp.erase();
  }

  // Set insertion point to the new block and add measurements
  builder.setInsertionPointToEnd(newBlock);
  auto measTy = quake::MeasureType::get(builder.getContext());
  for (auto &[index, alloca] : llvm::enumerate(allocations)) {
    if (isa<quake::VeqType>(alloca.getType())) {
      auto stdvecTy = cudaq::cc::StdvecType::get(measTy);
      builder.create<quake::MzOp>(loc, stdvecTy,
                                  ValueRange{alloca.getResult()});
    } else {
      builder.create<quake::MzOp>(loc, measTy, alloca.getResult());
    }
  }

  // Add the final return using block arguments
  builder.create<func::ReturnOp>(loc, newBlock->getArguments());

  return success();
}

struct AddMeasurementsPass
    : public cudaq::opt::impl::AddMeasurementsBase<AddMeasurementsPass> {
  using AddMeasurementsBase::AddMeasurementsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func || func.empty())
      return;

    if (!func->hasAttr(cudaq::entryPointAttrName))
      return;

    /// NOTE: Having a conditional on a measurement indicates that a measurement
    /// is present, however, it does not guarantee that all the allocated qubits
    /// are measured.
    if (auto boolAttr = func->getAttr("qubitMeasurementFeedback")
                            .dyn_cast_or_null<mlir::BoolAttr>()) {
      if (boolAttr.getValue())
        return;
    }

    // Check if the function has any measurement operations, if yes, we don't do
    // anything. If not, then check if the function has any qubit allocations,
    // if yes, then we want to add measurements to it.
    /// NOTE: Having an explicit measurement does not guarantee that all the
    /// allocated qubits are measured.
    Analysis analysis(func);
    if (analysis.hasMeasurement || !analysis.hasQubitAlloca())
      return;

    LLVM_DEBUG(llvm::dbgs() << "Before adding measurements:\n" << *func);
    if (failed(addMeasurements(func, analysis.allocations, analysis.returns)))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After adding measurements:\n" << *func);
  }
};
} // namespace
