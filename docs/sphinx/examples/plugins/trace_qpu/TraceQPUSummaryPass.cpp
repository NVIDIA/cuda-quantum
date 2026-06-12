/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/Plugin.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class TraceQPUSummaryPass
    : public PassWrapper<TraceQPUSummaryPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TraceQPUSummaryPass)

  StringRef getArgument() const override { return "trace-qpu-summary"; }

  StringRef getDescription() const override {
    return "Print a compact summary of quantum operations in a kernel.";
  }

  void runOnOperation() override {
    auto function = getOperation();
    unsigned quakeOps = 0;
    unsigned measurements = 0;

    function.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (!name.starts_with("quake."))
        return;

      ++quakeOps;
      if (name == "quake.mx" || name == "quake.my" || name == "quake.mz")
        ++measurements;
    });

    if (!quakeOps)
      return;

    llvm::errs() << "trace-qpu-summary: kernel=" << function.getSymName()
                 << " quake_ops=" << quakeOps
                 << " measurements=" << measurements << "\n";
  }
};
} // namespace

CUDAQ_REGISTER_MLIR_PASS(TraceQPUSummaryPass)
