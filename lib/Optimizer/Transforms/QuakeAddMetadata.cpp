/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

#define DEBUG_TYPE "add-metadata"

namespace {

/// Define a type to contain the Quake Function Metadata
struct QuakeMetadata {
  bool hasConditionalsOnMeasure = false;

  // If the following flag is set, it means we've detected quantum to classical
  // back to quantum data-flow in the kernel. This could be a problem for
  // quantum hardware.
  bool hasQuantumDataflowViaClassical = false;

  // If the following flag is set, this pass was run early enough that function
  // calls have not been inlined and we have quantum computation that excapes
  // the kernel. We flag this condition pessimistically, since we may not know
  // what the called function will do.
  bool hasUnexpectedCalls = false;
};
} // namespace

static cudaq::cc::AllocaOp seekAllocaFrom(Value v);

static cudaq::cc::AllocaOp seekAllocaFrom(Operation *op) {
  if (!op)
    return {};
  if (auto alloca = dyn_cast<cudaq::cc::AllocaOp>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Found alloca " << alloca << " from store.\n");
    return alloca;
  }
  if (auto cp = dyn_cast<cudaq::cc::ComputePtrOp>(op))
    return seekAllocaFrom(cp.getBase());
  if (auto castOp = dyn_cast<cudaq::cc::CastOp>(op))
    if (isa<cudaq::cc::PointerType>(castOp.getOperand().getType()))
      return seekAllocaFrom(castOp.getValue());
  return {};
}

static cudaq::cc::AllocaOp seekAllocaFrom(Value v) {
  if (!v)
    return {};
  return seekAllocaFrom(v.getDefiningOp());
}

/// If the operation is a Measurement, check if its qubits are used in a
/// subsequent reset operation, return true if so.
static bool checkIsMeasureAndReset(quake::MeasurementInterface mxOp) {
  if (mxOp.getOptionalRegisterName())
    for (Value measuredQubit : mxOp.getTargets())
      for (Operation *user : measuredQubit.getUsers())
        if (isa_and_present<quake::ResetOp>(user))
          return true;
  return false;
}

namespace {
/// We'll define a type mapping a Quake Function to its metadata
using QuakeFunctionInfo = DenseMap<Operation *, QuakeMetadata>;

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

    LLVM_DEBUG(llvm::dbgs()
               << "Function to analyze: " << funcOp.getName() << '\n');
    QuakeMetadata data;
    SmallPtrSet<Operation *, 8> dirtySet;
    funcOp->walk([&](quake::DiscriminateOp disc) {
      dirtySet.insert(disc.getOperation());
    });

    // If there are no discriminate ops, we can stop.
    if (dirtySet.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Found no discriminate ops\n");
      infoMap.insert({operation, data});
      return;
    }

    // Iterate to a fix-point, collecting all Ops that are data-dependent on the
    // result of a quantum computation via a quake.discriminate Op. All quantum
    // computations must, by definition use a quake.discriminate Op to convert
    // quantum values to classical values. This may terminate early if the set
    // of dirty ops collected includes a branch (control flow) or a quantum
    // operation (data flow).
    SmallVector<Operation *> keys{dirtySet.begin(), dirtySet.end()};
    auto addUser = [&](Operation *user) {
      auto [iter, added] = dirtySet.insert(user);
      (void)iter;
      if (added) {
        LLVM_DEBUG(llvm::dbgs() << "op " << *user << " was added\n");
        keys.push_back(user);
      }
    };

    do {
      auto *op = keys.back();
      keys.pop_back();
      if (isa<cudaq::cc::IfOp, cudaq::cc::ConditionOp, cf::CondBranchOp,
              quake::MeasurementInterface, quake::OperatorInterface,
              quake::ApplyOp, cudaq::cc::CallCallableOp, func::CallOp,
              func::CallIndirectOp>(op)) {
        data.hasConditionalsOnMeasure = true;
        data.hasQuantumDataflowViaClassical =
            isa<quake::MeasurementInterface, quake::OperatorInterface>(op);
        data.hasUnexpectedCalls = isa<quake::ApplyOp, cudaq::cc::CallCallableOp,
                                      func::CallOp, func::CallIndirectOp>(op);
        LLVM_DEBUG(llvm::dbgs() << "FOUND: mid-circuit dependence!\n");
        break;
      }
      for (auto *user : op->getUsers()) {
        addUser(user);
        // NB: This chases a store back to an allocation. It is possible that
        // the store is to an aggregate and that some values in the aggregate
        // are quantum dirty but NOT the ones used in classical control- or
        // data-flow. This doesn't perform points-to analysis on the interior of
        // the object. The analysis may be overly conservative.
        if (auto storeOp = dyn_cast<cudaq::cc::StoreOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "store seen " << storeOp << '\n');
          if (auto alloca = seekAllocaFrom(storeOp.getPtrvalue())) {
            LLVM_DEBUG(llvm::dbgs() << "alloca seen " << alloca << '\n');
            addUser(alloca.getOperation());
          }
        }
      }
    } while (!keys.empty());

    if (data.hasConditionalsOnMeasure) {
      infoMap.insert({operation, data});
      return;
    }

    // We also want to be able to sample differently if we have no conditionals
    // but do have a mz to a classical register with a subsequent reset call.
    // Handles
    //   auto reg = mz(q);
    //   reset(q);
    // don't necessarily need conditional statements
    funcOp->walk([&](quake::MeasurementInterface meas) {
      // NB: checkIsMeasureAndReset does NOT check the order or any control
      // flow. We only know that a measurement and a reset acted on the same
      // SSA-value. This is overly conservative and possibly a lurking bug.
      if (checkIsMeasureAndReset(meas)) {
        LLVM_DEBUG(llvm::dbgs() << "reset and measure on same ref\n");
        data.hasConditionalsOnMeasure = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    infoMap.insert({operation, data});
  }

  QuakeFunctionInfo infoMap;
};

/// This pass will analyze Quake functions and attach metadata (as an MLIR
/// function attribute) for specific features.
class QuakeAddMetadataPass
    : public cudaq::opt::QuakeAddMetadataBase<QuakeAddMetadataPass> {
public:
  QuakeAddMetadataPass() = default;

  /// This analysis is most effective if factor-quantum-alloc and memtoreg
  /// have been run prior to this pass. If not, this pass may give false
  /// positives. expand-measurements and loop-unrolling may further reduce
  /// false positives.
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
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createQuakeAddMetadata() {
  return std::make_unique<QuakeAddMetadataPass>();
}
