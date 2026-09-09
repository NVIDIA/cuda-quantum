/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ERASECOMPILERGENERATEDEVINCE
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-implicit-output"

using namespace mlir;

namespace {

class EraseCompilerGeneratedEvincePass
    : public cudaq::opt::impl::EraseCompilerGeneratedEvinceBase<
          EraseCompilerGeneratedEvincePass> {
public:
  using EraseCompilerGeneratedEvinceBase::EraseCompilerGeneratedEvinceBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](cudaq::quake::EvinceOp evince) {
      if (!evince.getCompilerGenerated())
        return;
      // For each wire/cable result, replace uses with the corresponding arg.
      unsigned resultIdx = 0;
      for (Value arg : evince.getArgs())
        if (cudaq::quake::isLinearType(arg.getType()))
          evince.getOuts()[resultIdx++].replaceAllUsesWith(arg);
      evince->erase();
    });
  }
};
} // namespace
