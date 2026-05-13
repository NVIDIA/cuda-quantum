/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#ifdef CUDAQ_HAS_CLIFFORD_T_SYNTHESIS
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Circuit/Gate.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Support/Result.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"
#endif

namespace cudaq::opt {
#define GEN_PASS_DEF_CLIFFORDTSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "clifford-t-synthesis"

using namespace mlir;

namespace {

class CliffordTSynthesisPass
    : public cudaq::opt::impl::CliffordTSynthesisBase<CliffordTSynthesisPass> {
public:
  using CliffordTSynthesisBase::CliffordTSynthesisBase;

  void runOnOperation() override {
#ifndef CUDAQ_HAS_CLIFFORD_T_SYNTHESIS
    getOperation().emitError(
        "clifford-t-synthesis: CUDA-Q was built without the synthesis "
        "library (GMP/MPFR missing at configure time). Rebuild with "
        "libgmp-dev and libmpfr-dev installed.");
    signalPassFailure();
    return;
#else
    LLVM_DEBUG(llvm::dbgs()
               << "clifford-t-synthesis: epsilon=" << epsilon
               << " diophantine-timeout-ms=" << diophantineTimeoutMs
               << " factoring-timeout-ms=" << factoringTimeoutMs
               << " retry-count=" << retryCount << " on-dynamic-angle="
               << onDynamicAngle << " skip-below=" << skipBelow << '\n');
#endif
  }
};

} // namespace
