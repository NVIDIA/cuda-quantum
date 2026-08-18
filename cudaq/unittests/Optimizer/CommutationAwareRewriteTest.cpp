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
    Operation *endpoint =
        matcher.findNearest(anchor, [&](Operation *candidate) {
          return isa<cudaq::quake::HOp>(candidate) &&
                 matcher.haveSameOrderedQuantumOperands(anchor, candidate);
        });
    if (!endpoint)
      return failure();

    rewriter.replaceOp(endpoint, anchor->getOperands());
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
    matchedBefore = matcher.findNearest(anchor, reachesEndpoint);

    auto replacedConstant =
        operation.getParameters().front().getDefiningOp<arith::ConstantOp>();
    rewriter.setInsertionPoint(replacedConstant);
    auto replacement = arith::ConstantFloatOp::create(
        rewriter, operation.getLoc(), rewriter.getF64Type(),
        llvm::APFloat(2.0));
    rewriter.replaceOp(replacedConstant, replacement.getResult());

    matchedAfter = matcher.findNearest(anchor, reachesEndpoint);
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
       StopsBeforeEndpointWhenARequiredWirePathEnds) {
  auto module = parseModule(R"mlir(
    module {
      func.func @ended_path() {
        %control = quake.null_wire
        %target = quake.null_wire
        %controlled:2 = quake.x [%control] %target
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %candidate = quake.x %controlled#1
            : (!quake.wire) -> !quake.wire
        quake.sink %candidate : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto operators = getOperators(getFunction(*module, "ended_path"));
  ASSERT_EQ(operators.size(), 2u);
  unsigned predicateCalls = 0;
  Operation *match =
      driver.getMatcher().findNearest(operators[0], [&](Operation *candidate) {
        ++predicateCalls;
        return candidate == operators[1];
      });

  EXPECT_FALSE(match);
  EXPECT_EQ(predicateCalls, 0u);
}

TEST_F(CommutationAwareRewriteTest,
       UsesExactThreadingForAdjacentUnwrapRootedRewrites) {
  auto module = parseModule(R"mlir(
    module {
      func.func @adjacent_cancel() {
        %reference = quake.alloca !quake.ref
        %unwrapped = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        %h0 = quake.h %unwrapped : (!quake.wire) -> !quake.wire
        %h1 = quake.h %h0 : (!quake.wire) -> !quake.wire
        quake.wrap %h1 to %reference : !quake.wire, !quake.ref
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  auto cancelFunction = getFunction(*module, "adjacent_cancel");
  cudaq::opt::CommutationAwareRewriteDriver cancelDriver(context);
  cancelDriver.getPatterns().add<CancelHadamard>(&context,
                                                 cancelDriver.getMatcher());
  EXPECT_TRUE(succeeded(cancelDriver.run(cancelFunction.getBody())));
  EXPECT_TRUE(cancelFunction.getOps<cudaq::quake::HOp>().empty());
  EXPECT_EQ(cancelDriver.getStatistics().analysisBuilds, 0u);

  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(CommutationAwareRewriteTest,
       RejectsAdjacentRoleMismatchWithoutAnalysis) {
  auto module = parseModule(R"mlir(
    module {
      func.func @role_mismatch() {
        %control = quake.null_wire
        %target = quake.null_wire
        %x0:2 = quake.x [%control] %target
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %x1:2 = quake.x [%x0#1] %x0#0
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %x1#0 : !quake.wire
        quake.sink %x1#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto operators = getOperators(getFunction(*module, "role_mismatch"));
  ASSERT_EQ(operators.size(), 2u);
  EXPECT_FALSE(driver.getMatcher().haveSameOrderedQuantumOperands(
      operators[0], operators[1]));
  EXPECT_EQ(driver.getStatistics().analysisBuilds, 0u);
}

TEST_F(CommutationAwareRewriteTest, RejectsDirectRepeatedWireOperand) {
  // The first operator uses one SSA wire as both control and target. Exact
  // threading to the adjacent operator must not bypass operand validation.
  auto module = parseModule(R"mlir(
    module {
      func.func @repeated_wire(%q: !quake.wire) {
        %x0:2 = quake.x [%q] %q
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %x1:2 = quake.x [%x0#0] %x0#1
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %x1#0 : !quake.wire
        quake.sink %x1#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto operators = getOperators(getFunction(*module, "repeated_wire"));
  ASSERT_EQ(operators.size(), 2u);
  EXPECT_FALSE(
      driver.getMatcher().findNearest(operators[0], [&](Operation *candidate) {
        return candidate == operators[1];
      }));
  EXPECT_FALSE(driver.getMatcher().haveSameOrderedQuantumOperands(
      operators[0], operators[1]));
  EXPECT_EQ(driver.getStatistics().analysisBuilds, 1u);
}

TEST_F(CommutationAwareRewriteTest, RejectsDirectAliasedWireOperands) {
  // The control and target have different SSA values but borrow the same
  // wire-set identity. Only identity normalization exposes the duplicate.
  auto module = parseModule(R"mlir(
    module {
      quake.wire_set @wires[1]
      func.func @aliased_wires() {
        %q0 = quake.borrow_wire @wires[0] : !quake.wire
        %q1 = quake.borrow_wire @wires[0] : !quake.wire
        %x0:2 = quake.x [%q0] %q1
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %x1:2 = quake.x [%x0#0] %x0#1
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %x1#0 : !quake.wire
        quake.sink %x1#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto operators = getOperators(getFunction(*module, "aliased_wires"));
  ASSERT_EQ(operators.size(), 2u);
  EXPECT_FALSE(
      driver.getMatcher().findNearest(operators[0], [&](Operation *candidate) {
        return candidate == operators[1];
      }));
  EXPECT_FALSE(driver.getMatcher().haveSameOrderedQuantumOperands(
      operators[0], operators[1]));
  EXPECT_EQ(driver.getStatistics().analysisBuilds, 1u);
}

TEST_F(CommutationAwareRewriteTest,
       ForwardsListenerRejectsReuseAndInvalidatesChangedRelations) {
  auto module = parseModule(R"mlir(
    module {
      func.func @cancel() {
        %control = quake.null_wire
        %target = quake.null_wire
        %h0:2 = quake.h [%control] %target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %h1:2 = quake.h [%h0#0] %h0#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %h1#0 : !quake.wire
        quake.sink %h1#1 : !quake.wire
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
