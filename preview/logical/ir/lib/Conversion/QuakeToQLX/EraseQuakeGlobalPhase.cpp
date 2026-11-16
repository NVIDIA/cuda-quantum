/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qlx/Conversion/QuakeToQLXPasses.h"

#include "mlir/IR/BuiltinOps.h"

#ifdef QLX_HAS_CUDAQ_QUAKE
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#endif

namespace qlx {
#define GEN_PASS_DEF_ERASEQUAKEGLOBALPHASE
#include "qlx/Conversion/QuakeToQLXPasses.h.inc"
} // namespace qlx

using namespace mlir;

namespace {

class EraseQuakeGlobalPhasePass
    : public qlx::impl::EraseQuakeGlobalPhaseBase<EraseQuakeGlobalPhasePass> {
public:
  void runOnOperation() override {
#ifdef QLX_HAS_CUDAQ_QUAKE
    getOperation().walk([](cudaq::quake::PhaseOp phase) {
      if (!phase.getControls().empty())
        return;
      // In value form the single result is the anchor wire's successor. In
      // reference form there is no result.
      for (Value wire : phase.getWires())
        wire.replaceAllUsesWith(phase.getTarget());
      phase.erase();
    });
#else
    getOperation().emitError(
        "Quake phase cleanup was not enabled; enable QLX_USE_CUDAQ_SDK");
    signalPassFailure();
#endif
  }
};

} // namespace
