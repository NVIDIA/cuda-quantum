/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

/// @brief Define a type to contain the Quake Function Metadata
struct QuakeMetadata {
  bool hasConditionalsOnMeasure = false;
};

/// @brief We'll define a type mapping a Quake Function to its metadata
using QuakeFunctionInfo = DenseMap<Operation *, QuakeMetadata>;

/// @brief If the operation is a Measurement, check if its
/// qubits are used in a subsequent reset operation,
/// return true if so.
bool checkIsMeasureAndReset(Operation *op, QuakeMetadata &data) {
  if (auto mxOp = dyn_cast<quake::MeasurementInterface>(op))
    if (mxOp.getOptionalRegisterName())
      for (auto measuredQubit : mxOp.getTargets())
        for (auto user : measuredQubit.getUsers())
          if (isa<quake::ResetOp>(user)) {
            data.hasConditionalsOnMeasure = true;
            return true;
          }

  return false;
}

/// The analysis on an a Quake function which will attach
/// metadata under certain situations.
struct QuakeFunctionAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuakeFunctionAnalysis)

  QuakeFunctionAnalysis(Operation *op) { performAnalysis(op); }
  const QuakeFunctionInfo &getAnalysisInfo() const { return infoMap; }

private:
  // Scan the body of a function for ops that will lead to the
  // addition of metadata.
  void performAnalysis(Operation *operation) {
    auto funcOp = dyn_cast<func::FuncOp>(operation);
    if (!funcOp)
      return;

    QuakeMetadata data;
    funcOp->walk([&](Operation *op) {
      // Strategy:
      // Look for Measure Ops, if the return value is used by a StoreOp
      // then get the memref.alloca value. Any loads from that alloca
      // that are used by conditionals is what we are looking for.
      if (!isa<quake::MeasurementInterface>(op))
        return WalkResult::skip();

      // Get the return bit value
      Value returnBit = op->getResult(0);

      // If no users, then no conditional
      if (returnBit.getUsers().empty())
        return WalkResult::skip();

      // Loop over all users of the return bit
      for (auto user : returnBit.getUsers()) {

        // See if we are immediately used by an If stmt.
        // If not, we'll do some more work before we give up
        if (isa<cudaq::cc::IfOp, cf::CondBranchOp>(user)) {
          data.hasConditionalsOnMeasure = true;
          return WalkResult::interrupt();
        }

        // See if it is a store op, storing the bit to memory
        auto storeOp = dyn_cast_or_null<cudaq::cc::StoreOp>(user);
        if (!storeOp)
          return WalkResult::skip();

        // Get the alloca op that this store op operates on
        auto allocValue = storeOp.getOperand(1);
        if (auto cp = allocValue.getDefiningOp<cudaq::cc::ComputePtrOp>())
          allocValue = cp.getBase();

        if (auto allocaOp = allocValue.getDefiningOp<cudaq::cc::AllocaOp>()) {
          // Get the alloca users
          for (auto allocUser : allocaOp->getUsers()) {

            // Look for any future loads, and if that load is
            // used by a conditional statement
            if (auto load = dyn_cast<cudaq::cc::LoadOp>(allocUser)) {
              auto loadUser = *load->getUsers().begin();

              // Loaded Val could be used directly or by an Arith boolean
              // operation
              while (loadUser->getDialect()->getNamespace() == "arith") {
                auto res = loadUser->getResult(0);
                loadUser = *res.getUsers().begin();
              }

              // At this point we should be able to check if we are
              // being used by a conditional
              if (isa<cudaq::cc::IfOp, cf::CondBranchOp>(loadUser)) {
                data.hasConditionalsOnMeasure = true;
                return WalkResult::interrupt();
              }
            }
          }
        }
      }

      return WalkResult::advance();
    });

    if (!data.hasConditionalsOnMeasure) {
      // We also want to be able to sample differently
      // if we have no conditionals but do have a mz to a
      // classical register with a subsequent reset call.
      // Handle auto reg = mz(q); reset(q)
      // don't necessarily need conditional statements
      funcOp->walk([&](Operation *op) {
        if (!isa<quake::MeasurementInterface>(op))
          return WalkResult::skip();

        // Return true if Reset on measured qubit,
        // if so just drop out because we'll have the function
        // tagged no matter what
        if (checkIsMeasureAndReset(op, data))
          return WalkResult::interrupt();

        return WalkResult::advance();
      });
    }

    infoMap.insert({operation, data});
  }

  /// @brief The Quake Function metadata map
  QuakeFunctionInfo infoMap;
};

/// @brief This pass will analyze Quake functions and attach
/// metadata (as an MLIR function attribute) for specific features.
class QuakeAddMetadataPass
    : public cudaq::opt::QuakeAddMetadataBase<QuakeAddMetadataPass> {
public:
  QuakeAddMetadataPass() = default;

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!funcOp || funcOp.empty())
      return;

    // Create the analysis and extract the info
    const auto &analysis = getAnalysis<QuakeFunctionAnalysis>();
    const auto &funcAnalysisInfo = analysis.getAnalysisInfo();
    auto iter = funcAnalysisInfo.find(funcOp);
    assert(iter != funcAnalysisInfo.end());
    const auto &info = iter->second;

    // Did this function have conditionals on measures?
    if (info.hasConditionalsOnMeasure) {
      // if so, add a function attribute
      auto builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
      funcOp->setAttr("qubitMeasurementFeedback", builder.getBoolAttr(true));
    }

    // ... others in the future ...
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createQuakeAddMetadata() {
  return std::make_unique<QuakeAddMetadataPass>();
}
