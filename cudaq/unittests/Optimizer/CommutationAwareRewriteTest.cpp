/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Transforms/CommutationAwareRewrite.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <gtest/gtest.h>

using namespace mlir;

namespace {

class CommutationAwareRewriteTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<cudaq::quake::QuakeDialect>();
  }

  OwningOpRef<ModuleOp> parseModule(llvm::StringRef source) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (module && succeeded(verify(*module)))
      return module;
    return {};
  }

  static func::FuncOp getFunction(ModuleOp module, llvm::StringRef name) {
    auto function = module.lookupSymbol<func::FuncOp>(name);
    EXPECT_TRUE(function);
    return function;
  }

  static llvm::SmallVector<Operation *> getOperators(func::FuncOp function) {
    llvm::SmallVector<Operation *> operators;
    for (Operation &operation : function.getBody().front())
      if (isa<cudaq::quake::OperatorInterface>(operation))
        operators.push_back(&operation);
    return operators;
  }

  MLIRContext context;
};

class CancelHadamard : public OpRewritePattern<cudaq::quake::HOp> {
public:
  CancelHadamard(MLIRContext *context,
                 cudaq::opt::CommutationAwareRewriteMatcher &matcher)
      : OpRewritePattern(context), matcher(matcher) {}

  LogicalResult matchAndRewrite(cudaq::quake::HOp anchor,
                                PatternRewriter &rewriter) const override {
    auto match = matcher.findNearest(
        anchor, cudaq::opt::CommutationSearchDirection::Forward,
        [&](Operation *candidate) {
          return isa<cudaq::quake::HOp>(candidate) &&
                 matcher.haveSameOrderedQuantumOperands(anchor, candidate);
        });
    if (!match)
      return failure();

    rewriter.replaceOp(match->endpoint, anchor->getOperands());
    rewriter.eraseOp(anchor);
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
};

class ReplaceOperatorParameter : public OpRewritePattern<cudaq::quake::U3Op> {
public:
  ReplaceOperatorParameter(MLIRContext *context,
                           cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                           bool &matchedBefore, bool &matchedAfter)
      : OpRewritePattern(context), matcher(matcher),
        matchedBefore(matchedBefore), matchedAfter(matchedAfter) {}

  LogicalResult matchAndRewrite(cudaq::quake::U3Op operation,
                                PatternRewriter &rewriter) const override {
    Operation *anchor = operation.getTargets().front().getDefiningOp();
    if (matchedBefore || !isa_and_nonnull<cudaq::quake::U3Op>(anchor) ||
        !operation->getResult(0).hasOneUse())
      return failure();
    Operation *endpoint = *operation->getResult(0).getUsers().begin();
    if (!isa<cudaq::quake::HOp>(endpoint))
      return failure();
    auto reachesEndpoint = [endpoint](Operation *candidate) {
      return candidate == endpoint;
    };
    matchedBefore =
        matcher
            .findNearest(anchor,
                         cudaq::opt::CommutationSearchDirection::Forward,
                         reachesEndpoint)
            .has_value();

    auto replacedConstant =
        operation.getParameters().front().getDefiningOp<arith::ConstantOp>();
    rewriter.setInsertionPoint(replacedConstant);
    auto replacement = arith::ConstantFloatOp::create(
        rewriter, operation.getLoc(), rewriter.getF64Type(),
        llvm::APFloat(2.0));
    rewriter.replaceOp(replacedConstant, replacement.getResult());

    matchedAfter =
        matcher
            .findNearest(anchor,
                         cudaq::opt::CommutationSearchDirection::Forward,
                         reachesEndpoint)
            .has_value();
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  bool &matchedBefore;
  bool &matchedAfter;
};

struct EraseListener : public RewriterBase::Listener {
  void notifyOperationErased(Operation *) override { ++eraseCount; }
  unsigned eraseCount = 0;
};

TEST_F(CommutationAwareRewriteTest,
       ReportsBackwardCrossingsInDeterministicBlockOrder) {
  auto module = parseModule(R"mlir(
    module {
      func.func @same_axis(%a: f64, %b: f64, %c: f64, %d: f64) {
        %q = quake.null_wire
        %rz0 = quake.rz (%a) %q : (f64, !quake.wire) -> !quake.wire
        %rz1 = quake.rz (%b) %rz0 : (f64, !quake.wire) -> !quake.wire
        %rz2 = quake.rz (%c) %rz1 : (f64, !quake.wire) -> !quake.wire
        %rz3 = quake.rz (%d) %rz2 : (f64, !quake.wire) -> !quake.wire
        quake.sink %rz3 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto &matcher = driver.getMatcher();
  auto sameAxis = getOperators(getFunction(*module, "same_axis"));
  ASSERT_EQ(sameAxis.size(), 4u);
  auto match = matcher.findNearest(
      sameAxis[3], cudaq::opt::CommutationSearchDirection::Backward,
      [&](Operation *candidate) { return candidate == sameAxis[0]; });
  ASSERT_TRUE(match);
  EXPECT_EQ(match->endpoint, sameAxis[0]);
  EXPECT_EQ(match->crossed,
            (llvm::SmallVector<Operation *>{sameAxis[1], sameAxis[2]}));
}

TEST_F(CommutationAwareRewriteTest,
       MaintainsPublicContractsAcrossObservedRewrites) {
  auto module = parseModule(R"mlir(
    module {
      func.func @cancel() {
        %q = quake.null_wire
        %h0 = quake.h %q : (!quake.wire) -> !quake.wire
        %h1 = quake.h %h0 : (!quake.wire) -> !quake.wire
        quake.sink %h1 : !quake.wire
        return
      }
      func.func @classical_replacement() {
        %theta0 = arith.constant 1.0 : f64
        %theta1 = arith.constant 1.0 : f64
        %zero = arith.constant 0.0 : f64
        %q = quake.null_wire
        %u0 = quake.u3 (%theta0, %zero, %zero) %q : (f64, f64, f64, !quake.wire) -> !quake.wire
        %u1 = quake.u3 (%theta1, %zero, %zero) %u0 : (f64, f64, f64, !quake.wire) -> !quake.wire
        %h = quake.h %u1 : (!quake.wire) -> !quake.wire
        quake.sink %h : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "cancel");

  EraseListener listener;
  GreedyRewriteConfig config;
  config.setListener(&listener);
  cudaq::opt::CommutationAwareRewriteDriver driver(context, config);
  driver.getPatterns().add<CancelHadamard>(&context, driver.getMatcher());

  EXPECT_TRUE(succeeded(driver.run(function.getBody())));
  EXPECT_TRUE(function.getOps<cudaq::quake::HOp>().empty());
  EXPECT_GT(listener.eraseCount, 0u);
  auto statistics = driver.getStatistics();
  EXPECT_EQ(statistics.analysisBuilds, 1u);
  EXPECT_EQ(statistics.fallbackRebuilds, 0u);
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(failed(driver.run(function.getBody())));

  // A used classical replacement may change operator parameters. The first
  // search caches that equal U3 operations commute; after theta1 changes, the
  // same search must stop rather than return that stale relation.
  auto parameterFunction = getFunction(*module, "classical_replacement");
  GreedyRewriteConfig parameterConfig;
  parameterConfig.enableFolding(false).enableConstantCSE(false);
  cudaq::opt::CommutationAwareRewriteDriver parameterDriver(context,
                                                            parameterConfig);
  bool matchedBefore = false;
  bool matchedAfter = true;
  parameterDriver.getPatterns().add<ReplaceOperatorParameter>(
      &context, parameterDriver.getMatcher(), matchedBefore, matchedAfter);
  EXPECT_TRUE(succeeded(parameterDriver.run(parameterFunction.getBody())));
  EXPECT_TRUE(matchedBefore);
  EXPECT_FALSE(matchedAfter);
  auto parameterStatistics = parameterDriver.getStatistics();
  EXPECT_EQ(parameterStatistics.analysisBuilds, 2u);
  EXPECT_EQ(parameterStatistics.fallbackRebuilds, 1u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
