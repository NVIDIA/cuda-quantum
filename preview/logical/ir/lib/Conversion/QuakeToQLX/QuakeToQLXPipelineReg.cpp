/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qlx/Conversion/QuakeToQLXPasses.h"

#ifdef QLX_HAS_CUDAQ_QUAKE
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#endif

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace qlx {

#ifdef QLX_HAS_CUDAQ_QUAKE
namespace {
/// P0 has no global phase. Discard it only at this import boundary, after
/// controlled applications have been expanded by CUDA-Q.
struct DropGlobalPhasePass
    : mlir::PassWrapper<DropGlobalPhasePass,
                        mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropGlobalPhasePass)

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    getOperation().walk([&](cudaq::quake::PhaseOp phase) {
      if (!phase.getControls().empty())
        return;
      // The target is only an ordering anchor. Preserve its SSA successor.
      if (phase.getWires().empty())
        rewriter.eraseOp(phase);
      else
        rewriter.replaceOp(phase, phase.getTarget());
    });
  }
};
} // namespace
#endif

void registerPrepareQuakeForQLXPipeline() {
  static bool registered = []() {
    mlir::PassPipelineRegistration<>(
        "prepare-quake-for-qlx",
        "Finalize CUDA-Q Quake already lowered through "
        "convert-to-linear-values for QLX P0 conversion",
        [](mlir::OpPassManager &pm) {
          pm.addPass(qlx::createPruneDeadCCLoopCarries());
#ifdef QLX_HAS_CUDAQ_QUAKE
          pm.addNestedPass<mlir::func::FuncOp>(
              std::make_unique<DropGlobalPhasePass>());
#endif
        });
    return true;
  }();
  (void)registered;
}

} // namespace qlx
