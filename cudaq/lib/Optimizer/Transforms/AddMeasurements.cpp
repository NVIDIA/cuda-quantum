/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
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
      } else if (isa<cudaq::quake::NullWireOp, cudaq::quake::NullCableOp,
                     cudaq::quake::BorrowWireOp>(op)) {
        allocations.emplace_back(op);
      } else if (isa<cudaq::quake::SinkOp, cudaq::quake::DeallocOp>(op)) {
        // Record deallocations. These supercede return ops in precedence.
        deallocations.emplace_back(op);
      } else if (isa<cudaq::quake::EvinceOp>(op)) {
        // Record the evince ops. These are used instead of the deallocations
        // when the use-evince option is set.
        evinces.emplace_back(op);
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
  SmallVector<Operation *> evinces;
  SmallVector<func::ReturnOp> returns;

  bool hasEvinces() const { return !evinces.empty(); }
  bool hasQubitDeallocs() const { return !deallocations.empty(); }
  bool hasQubitAlloca() const { return !allocations.empty(); }
};

/// Add measurement operations in front of the terminal operations, which may
/// be deallocations, sinks, or evince ops. Each quantum operand is measured in
/// operand order. This transformation is written to work in either reference or
/// value semantics. There is no control-flow rewrite. We expect that an
/// analysis will be provided to determine terminal measurement operations.
LogicalResult addMeasurementsBefore(func::FuncOp funcOp,
                                    SmallVector<Operation *> &terminators) {
  auto ctx = funcOp.getContext();
  OpBuilder builder(ctx);

  auto measTy = cudaq::quake::MeasureType::get(builder.getContext());
  auto sequenceTy = cudaq::cc::SequenceType::get(measTy);
  for (auto *op : terminators) {
    auto loc = op->getLoc();
    builder.setInsertionPoint(op);
    for (auto [index, opnd] : llvm::enumerate(op->getOperands())) {
      auto ty = opnd.getType();
      if (!cudaq::quake::isQuantumType(ty))
        continue;
      if (cudaq::quake::isLinearType(ty)) {
        // Thread the measured value back into the operation.
        auto meas = cudaq::quake::MzOp::create(builder, loc,
                                               TypeRange{measTy, ty}, opnd);
        op->setOperand(index, meas.getResult(1));
      } else {
        auto resTy = isa<cudaq::quake::RefType>(ty) ? Type{measTy} : sequenceTy;
        cudaq::quake::MzOp::create(builder, loc, resTy, opnd);
      }
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
      auto sequenceTy = cudaq::cc::SequenceType::get(measTy);
      cudaq::quake::MzOp::create(builder, loc, sequenceTy,
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

/// Populate the `mapping_reorder_idx` attribute on \p funcOp.
///
/// The mapping pass places virtual qubit `v` on physical qubit
/// `mapping_v2p[v]`, and a backend returns the global register sorted by
/// physical qubit. This attribute records the permutation that puts those bits
/// back into qubit allocation order; it is handed to
/// `sample_result::reorder()` after execution.
///
/// The operands of the evince ops are in allocation order, so operand `i`
/// identifies virtual qubit `i`. Only call this when every evinced qubit was
/// measured: the attribute must have exactly one entry per bit in the global
/// register.
void setMappingReorderIdx(func::FuncOp funcOp,
                          const SmallVector<Operation *> &evinces) {
  auto mappingAttr =
      dyn_cast_if_present<ArrayAttr>(funcOp->getAttr("mapping_v2p"));
  if (!mappingAttr)
    return;

  // Flatten the evince operands into allocation order. A qubit can be evinced
  // more than once when the returns do not share an exit block, so keep only
  // the first occurrence of each.
  SetVector<Value> qubits;
  for (Operation *op : evinces)
    for (Value opnd : op->getOperands())
      if (cudaq::quake::isQuantumType(opnd.getType()))
        qubits.insert(opnd);
  if (qubits.empty() || qubits.size() > mappingAttr.size())
    return;

  // pair is <first=virtual, second=physical>
  using VirtPhyPairType = std::pair<std::size_t, std::size_t>;
  SmallVector<VirtPhyPairType> measuredQubits;
  measuredQubits.reserve(qubits.size());
  for (std::size_t v = 0; v < qubits.size(); ++v)
    measuredQubits.emplace_back(v, cast<IntegerAttr>(mappingAttr[v]).getInt());

  // Sort by physical qubit, which is the order the backend reports, then find
  // the permutation that restores virtual (allocation) order.
  llvm::sort(measuredQubits,
             [](const VirtPhyPairType &a, const VirtPhyPairType &b) {
               return a.second < b.second;
             });
  SmallVector<std::size_t> reorderIdx(measuredQubits.size());
  for (std::size_t ix = 0; auto &element : reorderIdx)
    element = ix++;
  llvm::sort(reorderIdx, [&](const std::size_t &i1, const std::size_t &i2) {
    return measuredQubits[i1].first < measuredQubits[i2].first;
  });

  OpBuilder builder(funcOp.getContext());
  SmallVector<Attribute> attrs(reorderIdx.size());
  for (std::size_t ix = 0; auto &element : attrs)
    element = IntegerAttr::get(builder.getIntegerType(64), reorderIdx[ix++]);
  funcOp->setAttr("mapping_reorder_idx", builder.getArrayAttr(attrs));
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
    if (useEvince && analysis.hasEvinces()) {
      if (failed(addMeasurementsBefore(func, analysis.evinces)))
        signalPassFailure();
      else
        setMappingReorderIdx(func, analysis.evinces);
    } else if (analysis.hasQubitDeallocs()) {
      if (failed(addMeasurementsBefore(func, analysis.deallocations)))
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
