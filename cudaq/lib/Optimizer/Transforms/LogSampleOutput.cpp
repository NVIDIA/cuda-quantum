/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOGSAMPLEOUTPUT
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

void insertReferenceMarker(OpBuilder &builder, Value target) {
  auto targetType = target.getType();
  StringRef callee;
  if (isa<cudaq::quake::RefType>(targetType)) {
    callee = cudaq::sampleOutputQubitMarker;
  } else if (auto veqType = dyn_cast<cudaq::quake::VeqType>(targetType)) {
    callee = cudaq::sampleOutputVeqMarker;
    if (veqType.hasSpecifiedSize())
      target = cudaq::quake::RelaxSizeOp::create(
          builder, target.getLoc(),
          cudaq::quake::VeqType::getUnsized(builder.getContext()), target);
  } else {
    return;
  }

  func::CallOp::create(builder, target.getLoc(), TypeRange{}, callee,
                       ValueRange{target});
}

void insertValueMarker(OpBuilder &builder, cudaq::quake::SinkOp sink) {
  Value target = sink.getTarget();
  StringRef callee;
  if (isa<cudaq::quake::WireType>(target.getType()))
    callee = cudaq::sampleOutputQubitMarker;
  else if (isa<cudaq::quake::CableType>(target.getType()))
    callee = cudaq::sampleOutputVeqMarker;
  else
    return;

  auto marker = cudaq::quake::CallByRefOp::create(
      builder, sink.getLoc(),
      FlatSymbolRefAttr::get(builder.getContext(), callee),
      TypeRange{target.getType()}, ValueRange{target});
  sink->setOperand(0, marker.getResult(0));
}

class LogSampleOutputPass
    : public cudaq::opt::impl::LogSampleOutputBase<LogSampleOutputPass> {
public:
  using LogSampleOutputBase::LogSampleOutputBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp> entryPoints;
    for (auto function : module.getOps<func::FuncOp>())
      if (!function.empty() && function->hasAttr(cudaq::entryPointAttrName))
        entryPoints.push_back(function);
    if (entryPoints.empty())
      return;

    cudaq::IRBuilder intrinsicBuilder(module.getContext());
    if (failed(intrinsicBuilder.loadIntrinsic(
            module, cudaq::sampleOutputQubitMarker)) ||
        failed(intrinsicBuilder.loadIntrinsic(module,
                                              cudaq::sampleOutputVeqMarker))) {
      signalPassFailure();
      return;
    }

    for (auto function : entryPoints) {
      SmallVector<Operation *> terminalCleanup;
      function.walk([&](func::ReturnOp returnOp) {
        for (Operation *operation = returnOp->getPrevNode(); operation;
             operation = operation->getPrevNode()) {
          if (!isa<cudaq::quake::DeallocOp, cudaq::quake::SinkOp>(operation))
            break;
          terminalCleanup.push_back(operation);
        }
      });

      for (Operation *operation : terminalCleanup) {
        if (auto dealloc = dyn_cast<cudaq::quake::DeallocOp>(operation)) {
          OpBuilder builder(dealloc);
          insertReferenceMarker(builder, dealloc.getReference());
        } else {
          auto sink = cast<cudaq::quake::SinkOp>(operation);
          OpBuilder builder(sink);
          insertValueMarker(builder, sink);
        }
      }
    }
  }
};

} // namespace
