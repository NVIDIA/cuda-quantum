/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "nlohmann/json.hpp"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_COMBINEMEASUREMENTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "combine-measurements"

using namespace mlir;

namespace {

class ExtendMeasurePattern : public OpRewritePattern<quake::MzOp> {
  using OutputNamesType =
      std::map<std::size_t, std::pair<std::size_t, std::string>>;

public:
  using OpRewritePattern::OpRewritePattern;

  explicit ExtendMeasurePattern(MLIRContext *ctx,
                                OutputNamesType &resultQubitVals)
      : OpRewritePattern(ctx), resultQubitVals(resultQubitVals) {}

  // Replace a pattern such as:
  // ```
  // func.func @kernel() attributes {"cudaq-entrypoint"} {
  //   %1 = ... : !quake.veq<4>
  //   %2 = quake.subveq %1, %c2, %c3 : (!quake.veq<4>, i32, i32) ->
  //        !quake.veq<2>
  //   %measOut = quake.mz %2 : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
  // }
  // ```
  // with:
  // ```
  // func.func @kernel() attributes {"cudaq-entrypoint", ["output_names",
  // "[[[0,[1,\22q0\22]],[1,[2,\22q1\22]]]]"]} {
  //   %1 = ... : !quake.veq<4>
  //   %measOut = quake.mz %1 : (!quake.veq<4>) -> !cc.stdvec<!quake.measure>
  // }
  // ```
  LogicalResult matchAndRewrite(quake::MzOp measure,
                                PatternRewriter &rewriter) const override {
    if (!measure.use_empty()) {
      measure.emitError("Only measures with no uses are supported");
      return failure();
    }

    auto veqOp = measure.getOperand(0);
    if (auto subveq = veqOp.getDefiningOp<quake::SubVeqOp>()) {
      Value lowOp = subveq.getLow();
      Value highOp = subveq.getHigh();
      if (auto constLow = lowOp.getDefiningOp<arith::ConstantIntOp>()) {
        auto low = static_cast<std::size_t>(constLow.value());
        if (auto constHigh = highOp.getDefiningOp<arith::ConstantIntOp>()) {
          auto high = static_cast<std::size_t>(constHigh.value());
          for (std::size_t i = low; i <= high; i++) {
            // Note: regname is ignored for OpenQasm2 targets
            resultQubitVals[i - low] = std::make_pair(i, std::to_string(i));
          }
          rewriter.replaceOpWithNewOp<quake::MzOp>(
              measure, measure.getResultTypes(), subveq.getVeq());
          return success();
        }
      }
    }

    return failure();
  }

private:
  OutputNamesType &resultQubitVals;
};

class CombineMeasurementsPass
    : public cudaq::opt::impl::CombineMeasurementsBase<
          CombineMeasurementsPass> {
  using OutputNamesType =
      std::map<std::size_t, std::pair<std::size_t, std::string>>;

public:
  using CombineMeasurementsBase::CombineMeasurementsBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    OpBuilder builder(func);

    LLVM_DEBUG(llvm::dbgs() << "Function before combining measurements:\n"
                            << func << "\n\n");

    RewritePatternSet patterns(ctx);
    OutputNamesType resultQubitVals;
    patterns.insert<ExtendMeasurePattern>(ctx, resultQubitVals);
    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns)))) {
      func.emitOpError("combining measurements failed");
      signalPassFailure();
    }

    // TODO: move measOut to the end of the func (see delayMeasurements?)
    if (!resultQubitVals.empty()) {
      nlohmann::json resultQubitJSON{resultQubitVals};
      func->setAttr(cudaq::opt::QIROutputNamesAttrName,
                    builder.getStringAttr(resultQubitJSON.dump()));
    }

    LLVM_DEBUG(llvm::dbgs() << "Function after combining measurements:\n"
                            << func << "\n\n");
  }
};
} // namespace
