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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nlohmann/json.hpp"

namespace cudaq::opt {
#define GEN_PASS_DEF_COMBINEMEASUREMENTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "combine-measurements"

using namespace mlir;

namespace {

struct Analysis {
  using OutputNamesType =
      std::map<std::size_t, std::pair<std::size_t, std::string>>;

  Analysis() = default;
  Analysis(const Analysis &) = delete;
  Analysis(Analysis &&) = delete;
  Analysis &operator=(const Analysis &) = delete;

  mlir::DenseMap<mlir::Value, std::size_t> measurements;
  OutputNamesType resultQubitVals;
  quake::MzOp lastMeasurement;

  bool empty() const { return measurements.empty(); }

  LogicalResult analyze(func::FuncOp func) {
    quake::AllocaOp qalloc;
    std::size_t currentOffset = 0;

    for (auto &block : func.getRegion()) {
      for (auto &op : block) {
        if (auto alloc = dyn_cast_or_null<quake::AllocaOp>(&op)) {
          if (qalloc)
            return op.emitError("Multiple qalloc statements found");

          qalloc = alloc;
        } else if (auto measure = dyn_cast_or_null<quake::MzOp>(&op)) {
          if (!measure.use_empty())
            return measure.emitError("Measurements with uses are not supported");

          auto veqOp = measure.getOperand(0);
          auto ty = veqOp.getType();

          std::size_t size = 0;
          if (auto veqTy = dyn_cast<quake::RefType>(ty))
            size = 1;
          else if (auto veqTy = dyn_cast<quake::VeqType>(ty)) {
            size = veqTy.getSize();
            if (size == 0)
              return op.emitError("Unknown measurement size");
          }

          measurements[measure.getMeasOut()] = currentOffset;
          lastMeasurement = measure;
          currentOffset += size;
        }
      }
    }

    return success();
  }
};

class ExtendQubitMeasurePattern : public OpRewritePattern<quake::MzOp> {
  using OutputNamesType =
      std::map<std::size_t, std::pair<std::size_t, std::string>>;

public:
  using OpRewritePattern::OpRewritePattern;

  explicit ExtendQubitMeasurePattern(MLIRContext *ctx, Analysis &analysis)
      : OpRewritePattern(ctx), analysis(analysis) {}

  // Replace a pattern such as:
  // ```
  // %0 = ...: !quake.veq<2>
  // %1 = quake.extract_ref %0[0] : (!quake.veq<2>) -> !quake.ref
  // %measOut = quake.mz %1 : (!quake.ref) -> !quake.measure
  // ```
  // with:
  // ```
  //   %1 = ... : !quake.veq<4>
  //   %measOut = quake.mz %1 : (!quake.veq<4>) -> !cc.stdvec<!quake.measure>
  // ```
  // And collect output names information:  `"[[[0,[1,"q0"]],[1,[2,"q1"]]]]"`
  LogicalResult matchAndRewrite(quake::MzOp measure,
                                PatternRewriter &rewriter) const override {

    auto veqOp = measure.getOperand(0);
    if (auto extract = veqOp.getDefiningOp<quake::ExtractRefOp>()) {
      auto veq = extract.getVeq();
      std::size_t idx;

      if (extract.hasConstantIndex())
        idx = extract.getConstantIndex();
      else if (auto cst =
                   extract.getIndex().getDefiningOp<arith::ConstantIntOp>())
        idx = static_cast<std::size_t>(cst.value());
      else
        return extract.emitError("Non-constant index in ExtractRef");

      auto offset = idx + analysis.measurements[measure.getMeasOut()];
      analysis.resultQubitVals[offset] =
          std::make_pair(idx, std::to_string(idx));

      auto resultType = cudaq::cc::StdvecType::get(measure.getType(0));
      if (measure == analysis.lastMeasurement)
        rewriter.replaceOpWithNewOp<quake::MzOp>(measure, TypeRange{resultType},
                                                 ValueRange{veq},
                                                 measure.getRegisterNameAttr());
      else
        rewriter.eraseOp(measure);
    }

    return failure();
  }

private:
  Analysis &analysis;
};

class ExtendVeqMeasurePattern : public OpRewritePattern<quake::MzOp> {
  using OutputNamesType =
      std::map<std::size_t, std::pair<std::size_t, std::string>>;

public:
  using OpRewritePattern::OpRewritePattern;

  explicit ExtendVeqMeasurePattern(MLIRContext *ctx, Analysis &analysis)
      : OpRewritePattern(ctx), analysis(analysis) {}

  // Replace a pattern such as:
  // ```
  //   %1 = ... : !quake.veq<4>
  //   %2 = quake.subveq %1, %c1, %c2 : (!quake.veq<4>, i32, i32) ->
  //        !quake.veq<2>
  //   %measOut = quake.mz %2 : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
  // ```
  // with:
  // ```
  //   %1 = ... : !quake.veq<4>
  //   %measOut = quake.mz %1 : (!quake.veq<4>) -> !cc.stdvec<!quake.measure>
  // ```
  // And collect output names information:  `"[[[0,[1,"q0"]],[1,[2,"q1"]]]]"`
  LogicalResult matchAndRewrite(quake::MzOp measure,
                                PatternRewriter &rewriter) const override {

    auto veqOp = measure.getOperand(0);
    if (auto subveq = veqOp.getDefiningOp<quake::SubVeqOp>()) {
      Value lowOp = subveq.getLow();
      Value highOp = subveq.getHigh();

      auto constLow = lowOp.getDefiningOp<arith::ConstantIntOp>();
      if (!constLow)
        return subveq.emitError("Non-constant low index in subveq");
      auto constHigh = highOp.getDefiningOp<arith::ConstantIntOp>();
      if (!constHigh)
        return subveq.emitError("Non-constant high index in subveq");

      auto low = static_cast<std::size_t>(constLow.value());
      auto high = static_cast<std::size_t>(constHigh.value());

      for (std::size_t i = low; i <= high; i++) {
        auto start = analysis.measurements[measure.getMeasOut()];
        auto offset = i - low + start;
        analysis.resultQubitVals[offset] = std::make_pair(i, std::to_string(i));
      }
      if (measure == analysis.lastMeasurement)
        rewriter.replaceOpWithNewOp<quake::MzOp>(
            measure, measure.getResultTypes(), ValueRange{subveq.getVeq()},
            measure.getRegisterNameAttr());
      else
        rewriter.eraseOp(measure);

      return success();
    }

    return failure();
  }

private:
  Analysis &analysis;
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

    // Analyze the function to find all qubit mappings.
    Analysis analysis;
    if (failed(analysis.analyze(func))) {
      func.emitOpError("Combining measurements failed");
      signalPassFailure();
    }

    if (analysis.empty())
      return;

    // Extend measurement into one last full measurement.
    RewritePatternSet patterns(ctx);
    patterns.insert<ExtendQubitMeasurePattern, ExtendVeqMeasurePattern>(
        ctx, analysis);
    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns)))) {
      func.emitOpError("Combining measurements failed");
      signalPassFailure();
    }

    // Add output names mapping attribute.
    if (!analysis.resultQubitVals.empty()) {
      nlohmann::json resultQubitJSON{analysis.resultQubitVals};
      func->setAttr(cudaq::opt::QIROutputNamesAttrName,
                    builder.getStringAttr(resultQubitJSON.dump()));
    }

    LLVM_DEBUG(llvm::dbgs() << "Function after combining measurements:\n"
                            << func << "\n\n");
  }
};
} // namespace
