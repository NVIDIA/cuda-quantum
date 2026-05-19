/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
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
/// the returns for redirection.
struct Analysis {
  Analysis() = default;

  explicit Analysis(func::FuncOp func) {
    func.walk([&](Operation *op) {
      if (op->hasTrait<cudaq::QuantumMeasure>()) {
        hasMeasurement = true;
        return WalkResult::interrupt();
      }
      if (auto alloc = dyn_cast<cudaq::quake::AllocaOp>(op)) {
        // Adjust the op if there is an InitState.
        if (alloc->hasOneUse()) {
          Operation *user = *alloc->getUsers().begin();
          if (isa<cudaq::quake::InitializeStateOp>(user))
            op = user;
        }
        allocations.emplace_back(op);
      } else if (isa<cudaq::quake::NullWireOp, cudaq::quake::NullCableOp>(op)) {
        allocations.emplace_back(op);
      } else if (isa<cudaq::quake::SinkOp, cudaq::quake::DeallocOp>(op)) {
        // Record deallocations. These supercede return ops in precedence.
        deallocations.emplace_back(op);
      } else if (isa<func::ReturnOp>(op)) {
        // Use returns if the deallocations are not present.
        returns.emplace_back(op);
      }
      return WalkResult::advance();
    });
  }

  bool hasMeasurement = false;
  SmallVector<Operation *> allocations;
  SmallVector<Operation *> deallocations;
  SmallVector<func::ReturnOp> returns;

  bool hasQubitDeallocs() const { return !deallocations.empty(); }
  bool hasQubitAlloca() const { return !allocations.empty(); }
};

/// Add measurement operations before the deallocations. This transformation is
/// written to work in either reference or value semantics. There is no
/// control-flow rewrite. We expect that an analysis will be provided to
/// determine terminal measurement operations.
LogicalResult addDeallocMeasurements(func::FuncOp funcOp,
                                     SmallVector<Operation *> &deallocations) {
  auto ctx = funcOp.getContext();
  OpBuilder builder(ctx);

  auto measTy = cudaq::quake::MeasureType::get(builder.getContext());
  auto stdvecTy = cudaq::cc::StdvecType::get(measTy);
  for (auto *op : deallocations) {
    if (auto dealloc = dyn_cast<cudaq::quake::DeallocOp>(op)) {
      auto loc = dealloc.getLoc();
      builder.setInsertionPoint(dealloc);
      auto resTy = [&]() -> Type {
        if (isa<cudaq::quake::RefType>(dealloc.getReference().getType()))
          return measTy;
        return stdvecTy;
      }();
      cudaq::quake::MzOp::create(builder, loc, resTy, dealloc.getReference());
    } else {
      auto sink = cast<cudaq::quake::SinkOp>(op);
      auto loc = sink.getLoc();
      builder.setInsertionPoint(sink);
      auto meas = cudaq::quake::MzOp::create(
          builder, loc, TypeRange{measTy, sink.getTarget().getType()},
          sink.getTarget());
      cudaq::quake::SinkOp::create(builder, loc, TypeRange{},
                                   meas.getResult(1));
      sink->dropAllReferences();
      sink.erase();
    }
  }
  return success();
}

/// Add measurement operations for all allocated qubits in a function.
/// This transformation creates a new block at the end of the function,
/// redirects all return operations to branch to this block, adds `quake.mz`
/// measurement operations for each qubit allocation, and adds a final return.
/// For vector allocations, the measurements are collected into a vector of
/// measurement results.
LogicalResult
addReturnMeasurements(func::FuncOp funcOp,
                      SmallVector<Operation *> &allocations,
                      const SmallVector<func::ReturnOp> &returnsToReplace) {
  auto ctx = funcOp.getContext();
  OpBuilder builder(ctx);

  // Create a new block at the end of the function.
  Block *newBlock = funcOp.addBlock();

  // Add block arguments for return values if the function returns anything
  ArrayRef<Type> returnTypes = funcOp.getFunctionType().getResults();
  if (!returnTypes.empty()) {
    auto loc = funcOp.getLoc();
    SmallVector<Location> argLocs(returnTypes.size(), loc);
    newBlock->addArguments(returnTypes, argLocs);
  }

  // Replace every func.return in the function with a branch to the new block.
  for (auto returnOp : returnsToReplace) {
    OpBuilder builder(returnOp);
    cf::BranchOp::create(builder, returnOp.getLoc(), newBlock,
                         returnOp.getOperands());
    returnOp.erase();
  }

  // Set insertion point to the new block and add measurements
  builder.setInsertionPointToEnd(newBlock);
  auto measTy = cudaq::quake::MeasureType::get(builder.getContext());
  for (auto [index, alloca] : llvm::enumerate(allocations)) {
    Type allocTy = alloca->getResult(0).getType();
    auto loc = alloca->getLoc();
    if (isa<cudaq::quake::VeqType>(allocTy)) {
      auto stdvecTy = cudaq::cc::StdvecType::get(measTy);
      cudaq::quake::MzOp::create(builder, loc, stdvecTy,
                                 ValueRange{alloca->getResult(0)});
    } else if (isa<cudaq::quake::RefType>(allocTy)) {
      auto val = alloca->getResult(0);
      cudaq::quake::MzOp::create(builder, loc, measTy, val);
    } else {
      return failure();
    }
  }

  // Add the final return using block arguments
  auto loc = funcOp.getLoc();
  func::ReturnOp::create(builder, loc, newBlock->getArguments());
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
    if (auto boolAttr = dyn_cast_if_present<mlir::BoolAttr>(
            func->getAttr("qubitMeasurementFeedback"))) {
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
    if (analysis.hasQubitDeallocs()) {
      if (failed(addDeallocMeasurements(func, analysis.deallocations)))
        signalPassFailure();
    } else {
      if (failed(addReturnMeasurements(func, analysis.allocations,
                                       analysis.returns)))
        signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "After adding measurements:\n" << *func);
  }
};
} // namespace
