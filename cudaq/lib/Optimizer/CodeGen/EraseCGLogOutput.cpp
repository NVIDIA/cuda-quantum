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
#define GEN_PASS_DEF_ERASECOMPILERGENERATEDLOGOUTPUT
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-implicit-output"

using namespace mlir;

namespace {

class EraseCompilerGeneratedLogOutputPass
    : public cudaq::opt::impl::EraseCompilerGeneratedLogOutputBase<
          EraseCompilerGeneratedLogOutputPass> {
public:
  using EraseCompilerGeneratedLogOutputBase::
      EraseCompilerGeneratedLogOutputBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](cudaq::quake::LogOutputOp logOut) {
      if (!logOut.getCompilerGenerated())
        return;
      // For each wire/cable result, replace uses with the corresponding arg.
      unsigned resultIdx = 0;
      for (Value arg : logOut.getArgs())
        if (cudaq::quake::isLinearType(arg.getType()))
          logOut.getOuts()[resultIdx++].replaceAllUsesWith(arg);
      logOut->erase();
    });
  }
};
} // namespace
