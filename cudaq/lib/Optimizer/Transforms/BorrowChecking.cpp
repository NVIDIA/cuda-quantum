/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Analysis/WireBorrowAnalysis.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_BORROWCHECKING
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "borrow-checking"

using namespace mlir;

namespace {
using cudaq::opt::WireBorrowAnalysis;

class BorrowCheckingPass
    : public cudaq::opt::impl::BorrowCheckingBase<BorrowCheckingPass> {
public:
  using BorrowCheckingBase::BorrowCheckingBase;

  void runOnOperation() override {
    bool failed = false;
    getOperation()->walk([&](func::FuncOp func) {
      WireBorrowAnalysis analysis(func);
      for (auto &conflict : analysis.getConflicts())
        failed |= report(conflict);
      if (warnOnPossible)
        for (Operation *ret : analysis.getUnresolvedReturns())
          ret->emitWarning("cannot determine which wire is returned; borrow "
                           "checking is imprecise here");
    });
    if (failed)
      signalPassFailure();
  }

private:
  /// Emit a diagnostic for \p c. Return true if the pass must fail.
  bool report(const WireBorrowAnalysis::Conflict &c) {
    using Certainty = WireBorrowAnalysis::Certainty;
    const bool definite = c.certainty == Certainty::Definite;
    if (!definite && !warnOnPossible)
      return false;
    const bool isError = definite && raiseFailure;

    std::string wire = ("@" + c.setName + "[" + Twine(c.identity) + "]").str();
    std::string msg;
    if (c.kind == WireBorrowAnalysis::Kind::DoubleBorrow)
      msg = "wire " + wire +
            (definite ? " is borrowed while it is already borrowed"
                      : " may be borrowed while it is already borrowed");
    else
      msg = "wire " + wire +
            (definite ? " is borrowed but never returned"
                      : " may be borrowed but never returned");

    InFlightDiagnostic diag =
        isError ? c.op->emitError(msg) : c.op->emitWarning(msg);
    if (c.kind == WireBorrowAnalysis::Kind::DoubleBorrow)
      for (Operation *other : c.otherBorrows)
        diag.attachNote(other->getLoc()) << "wire is borrowed here";
    return isError;
  }
};
} // namespace
