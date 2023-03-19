/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/CodeGen/DecompositionGateSetMapping.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
class R1DecompositionGateSetMapper
    : public cudaq::opt::R1DecompositionGateSetMapperBase<
          R1DecompositionGateSetMapper> {
public:
  void runOnOperation() override {
    auto *context = getOperation().getContext();
    RewritePatternSet patterns(context);
    patterns.insert<QtnmR1Pattern>(context);
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, quake::QuakeDialect>();
    target.addDynamicallyLegalOp<quake::R1Op>(
        [](quake::R1Op rop) { return rop.getControls().size() == 0; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class RnDecompositionGateSetMapper
    : public cudaq::opt::RnDecompositionGateSetMapperBase<
          RnDecompositionGateSetMapper> {
public:
  void runOnOperation() override {
    auto *context = getOperation().getContext();
    RewritePatternSet patterns(context);
    patterns.insert<QtnmRxPattern, QtnmRyPattern, QtnmRzPattern>(context);
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, quake::QuakeDialect>();
    target.addDynamicallyLegalOp<quake::RxOp>(
        [](quake::RxOp rop) { return rop.getControls().size() == 0; });
    target.addDynamicallyLegalOp<quake::RyOp>(
        [](quake::RyOp rop) { return rop.getControls().size() == 0; });
    target.addDynamicallyLegalOp<quake::RzOp>(
        [](quake::RzOp rop) { return rop.getControls().size() == 0; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class QuantinuumGateSetMapper
    : public cudaq::opt::QuantinuumGateSetMapperBase<QuantinuumGateSetMapper> {
public:
  void runOnOperation() override {
    auto *context = getOperation().getContext();
    RewritePatternSet patterns(context);
    patterns.insert<QtnmRxPattern, QtnmRyPattern, QtnmRzPattern, QtnmR1Pattern>(
        context);
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, quake::QuakeDialect>();
    target.addDynamicallyLegalOp<quake::RxOp>(
        [](quake::RxOp rop) { return rop.getControls().size() == 0; });
    target.addDynamicallyLegalOp<quake::RyOp>(
        [](quake::RyOp rop) { return rop.getControls().size() == 0; });
    target.addDynamicallyLegalOp<quake::RzOp>(
        [](quake::RzOp rop) { return rop.getControls().size() == 0; });
    target.addDynamicallyLegalOp<quake::R1Op>(
        [](quake::R1Op rop) { return rop.getControls().size() == 0; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createR1DecompositionGateSetMapping() {
  return std::make_unique<R1DecompositionGateSetMapper>();
}
std::unique_ptr<Pass> cudaq::opt::createRnDecompositionGateSetMapping() {
  return std::make_unique<RnDecompositionGateSetMapper>();
}
std::unique_ptr<Pass> cudaq::opt::createQuantinuumGateSetMapping() {
  return std::make_unique<QuantinuumGateSetMapper>();
}
