/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_QIRINSERTARRAYRECORD
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "qir-insert-array-record"

using namespace mlir;

// Inserts a QIR array record output call to declare measurement result storage.
// Before recording multiple measurement results, QIR requires an array
// recording call to declare the output size and type label. This is necessary
// for the `sample` API, which returns a vector of measurement results. The
// call is inserted before the first result recording call. The label string
// has the format `array<i1 x N>` where N is the total number of measurement
// results. The generated call is: `__quantum__rt__array_record_output(N,
// label)`.
static LogicalResult insertArrayRecordingCall(OpBuilder &builder,
                                              mlir::Location loc,
                                              size_t resultCount) {
  if (resultCount == 0)
    return success();
  // Create the label string: `array<i1 x N>`
  std::string labelStr = "array<i1 x " + std::to_string(resultCount) + ">";
  auto strLitTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(
      builder.getContext(), builder.getI8Type(), labelStr.size() + 1));
  Value lit = builder.create<cudaq::cc::CreateStringLiteralOp>(
      loc, strLitTy, builder.getStringAttr(labelStr));
  auto i8PtrTy = cudaq::cc::PointerType::get(builder.getI8Type());
  Value label = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, lit);
  Value size = builder.create<arith::ConstantIntOp>(loc, resultCount, 64);
  builder.create<func::CallOp>(loc, TypeRange{},
                               cudaq::opt::QIRArrayRecordOutput,
                               ArrayRef<Value>{size, label});
  return success();
}

namespace {
struct QirInsertArrayRecordPass
    : public cudaq::opt::impl::QirInsertArrayRecordBase<
          QirInsertArrayRecordPass> {

  using QirInsertArrayRecordBase::QirInsertArrayRecordBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (!funcOp || funcOp.empty() ||
          !funcOp->hasAttr(cudaq::entryPointAttrName) ||
          funcOp->hasAttr(cudaq::runtime::enableCudaqRun))
        continue;

      SmallVector<func::CallOp> recordOutputCalls;
      bool arrayRecordExists = false;

      funcOp.walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == cudaq::opt::QIRArrayRecordOutput) {
          arrayRecordExists = true;
          return;
        }
        if (callOp.getCallee() == cudaq::opt::QIRRecordOutput)
          recordOutputCalls.push_back(callOp);
      });

      if (arrayRecordExists || recordOutputCalls.empty())
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Before adding array recording call:\n"
                              << *funcOp);
      // Add the declaration of array recording call to the module

      auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
      if (failed(irBuilder.loadIntrinsic(module,
                                         cudaq::opt::QIRArrayRecordOutput))) {
        return signalPassFailure();
      }

      // Insert array record before first result result recording call.
      OpBuilder builder(recordOutputCalls.front());
      if (failed(insertArrayRecordingCall(builder, funcOp.getLoc(),
                                          recordOutputCalls.size())))
        return signalPassFailure();
      LLVM_DEBUG(llvm::dbgs() << "After adding array recording call:\n"
                              << *funcOp);
    }
  }
};
} // namespace
