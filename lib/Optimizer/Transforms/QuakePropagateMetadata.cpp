/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/CallGraph.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKEPROPAGATEMETADATA
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "quake-propagate-metadata"

using namespace mlir;

namespace {

/// This pass will analyze Quake functions and attach metadata (as an MLIR
/// function attribute) for specific features.
class QuakePropagateMetadataPass
    : public cudaq::opt::impl::QuakePropagateMetadataBase<
          QuakePropagateMetadataPass> {
protected:
public:
  using QuakePropagateMetadataBase::QuakePropagateMetadataBase;

  /// This analysis is most effective if factor-quantum-alloc and memtoreg
  /// have been run prior to this pass. If not, this pass may give false
  /// positives. expand-measurements and loop-unrolling may further reduce
  /// false positives.
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    /// NOTE: If the module has an occurrence of `quake.apply` then the step to
    /// build call graph fails. Hence, we skip the pass in such cases.
    if (moduleOp.walk([](quake::ApplyOp op) { return WalkResult::interrupt(); })
            .wasInterrupted()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Skipping `QuakePropagateMetadataPass` due to `quake.apply`\n");
      return;
    }
    // Build the call graph of the module
    const mlir::CallGraph callGraph(moduleOp);
    // Use PostOrderTraversal to get the ordered list of FuncOps and a map of
    // callers for each function.
    llvm::ReversePostOrderTraversal<const mlir::CallGraph *> rpot(&callGraph);
    llvm::SmallVector<func::FuncOp> funcOps;
    llvm::DenseMap<func::FuncOp, llvm::SmallVector<func::FuncOp>> callerOps;
    for (auto &node : rpot) {
      if (node->isExternal())
        continue;

      auto *callableRegion = node->getCallableRegion();
      if (auto callerFnOp = callableRegion->getParentOfType<func::FuncOp>()) {
        funcOps.insert(funcOps.begin(), callerFnOp);

        for (auto it = node->begin(); it != node->end(); it++) {
          auto edge = *it;
          auto callee = edge.getTarget();
          if (callee->isExternal())
            continue;

          auto *calleeCallableRegion = callee->getCallableRegion();
          if (auto calleeFnOp =
                  calleeCallableRegion->getParentOfType<func::FuncOp>()) {

            if (callerOps.find(calleeFnOp) == callerOps.end()) {
              SmallVector<func::FuncOp> v;
              callerOps[calleeFnOp] = v;
            }
            callerOps[calleeFnOp].emplace_back(callerFnOp);
          }
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Module before propagating metadata:\n"
                            << moduleOp << "\n\n");

    for (auto &callee : funcOps) {
      auto callers = callerOps[callee];
      LLVM_DEBUG(llvm::dbgs()
                 << "Visiting callee: " << callee.getName() << "\n\n");
      for (auto caller : callers) {

        LLVM_DEBUG(llvm::dbgs() << "  Caller: " << caller.getName() << "\n\n");
        if (auto boolAttr = callee->getAttr("qubitMeasurementFeedback")
                                .dyn_cast_or_null<mlir::BoolAttr>()) {
          if (boolAttr.getValue()) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  Propagating qubitMeasurementFeedback attr: "
                       << boolAttr << "\n");
            caller->setAttr("qubitMeasurementFeedback", boolAttr);
          }
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Module after propagating metadata:\n"
                            << moduleOp << "\n\n");
  }
};

} // namespace
