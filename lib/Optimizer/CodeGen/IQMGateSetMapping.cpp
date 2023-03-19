/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/CodeGen/IQMGateSetMapping.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
class IQMGateSetMapper
    : public cudaq::opt::IQMGateSetMapperBase<IQMGateSetMapper> {
public:
  void runOnOperation() override {
    auto *context = getOperation().getContext();
    RewritePatternSet patterns(context);
    patterns.insert<IQMRxPattern, IQMRyPattern, IQMRzPattern, IQMHPattern,
                    IQMXPattern, IQMYPattern, IQMZPattern, IQMSPattern,
                    IQMTPattern, IQMR1Pattern, IQMXctrlPattern>(context);
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, quake::QuakeDialect>();
    target.addIllegalOp<quake::RxOp, quake::RyOp, quake::RzOp, quake::HOp,
                        quake::XOp, quake::YOp, quake::SOp, quake::TOp,
                        quake::R1Op>();
    target.addDynamicallyLegalOp<quake::ZOp>(
        [](quake::ZOp z) { return z.getControls().size() == 1u; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createIQMGateSetMapping() {
  return std::make_unique<IQMGateSetMapper>();
}
