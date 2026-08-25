/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_UPDATEREGISTERNAMES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "update-register-names"

using namespace mlir;

namespace {
/// After unrolling the loops, there may be duplicate registerName attributes in
/// use. This pass will assign them unique names by appending a counter.
class UpdateRegisterNamesPass
    : public cudaq::opt::impl::UpdateRegisterNamesBase<
          UpdateRegisterNamesPass> {
public:
  using UpdateRegisterNamesBase::UpdateRegisterNamesBase;

  void runOnOperation() override {
    auto *mod = getOperation();

    // First save the op's that contain a registerName attribute
    DenseMap<StringRef, SmallVector<Operation *>> regOps;
    mod->walk([&](mlir::Operation *walkOp) {
      if (auto prevAttr = walkOp->getAttr("registerName")) {
        auto registerName = cast<StringAttr>(prevAttr).getValue();
        regOps[registerName].push_back(walkOp);
      }
      return WalkResult::advance();
    });

    // Now apply new labels, appending a counter if necessary
    for (auto &[registerName, opVec] : regOps) {
      if (opVec.size() == 1)
        continue; // don't rename individual qubit measurements
      auto strLen = std::to_string(opVec.size()).size();
      int bit = 0;
      for (auto &regOp : opVec)
        if (auto prevAttr = regOp->getAttr("registerName")) {
          auto suffix = std::to_string(bit++);
          if (suffix.size() < strLen)
            suffix = std::string(strLen - suffix.size(), '0') + suffix;
          // Note Quantinuum can't support a ":" delimiter, so use '%'
          auto newAttr = OpBuilder(&getContext())
                             .getStringAttr(registerName + "%" + suffix);
          regOp->setAttr("registerName", newAttr);
        }
    }
  }
};
} // namespace
