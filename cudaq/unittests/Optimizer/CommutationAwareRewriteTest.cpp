/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Transforms/CommutationAwareRewrite.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
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
    context.loadDialect<cudaq::cc::CCDialect>();
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

  LogicalResult matchAndRewrite(cudaq::quake::HOp later,
                                PatternRewriter &rewriter) const override {
    Operation *earlier = matcher.find_nearest(later, [&](Operation *candidate) {
      return isa<cudaq::quake::HOp>(candidate) &&
             matcher.have_same_ordered_quantum_operands(later, candidate);
    });
    if (!earlier)
      return failure();

    rewriter.replaceOp(later, earlier->getOperands());
    rewriter.eraseOp(earlier);
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
};

enum class ParameterChange { Replacement, InPlaceMutation };

class ChangeOperatorParameter : public OpRewritePattern<cudaq::quake::U3Op> {
public:
  ChangeOperatorParameter(
      MLIRContext *context, cudaq::opt::CommutationAwareRewriteMatcher &matcher,
      bool &matchedBefore, bool &matchedAfter, ParameterChange change,
      Operation *primedLater = nullptr, Operation *primedEarlier = nullptr,
      bool *primedMatchedBefore = nullptr, bool *primedMatchedAfter = nullptr)
      : OpRewritePattern(context), matcher(matcher),
        matchedBefore(matchedBefore), matchedAfter(matchedAfter),
        change(change), primedLater(primedLater), primedEarlier(primedEarlier),
        primedMatchedBefore(primedMatchedBefore),
        primedMatchedAfter(primedMatchedAfter) {}

  LogicalResult matchAndRewrite(cudaq::quake::U3Op operation,
                                PatternRewriter &rewriter) const override {
    Operation *earlier = operation.getTargets().front().getDefiningOp();
    if (matchedBefore || !isa_and_nonnull<cudaq::quake::U3Op>(earlier) ||
        !operation->getResult(0).hasOneUse())
      return failure();
    Operation *later = *operation->getResult(0).getUsers().begin();
    if (!isa<cudaq::quake::U3Op>(later))
      return failure();
    auto reachesEarlier = [earlier](Operation *candidate) {
      return candidate == earlier;
    };
    if (primedLater &&
        (!primedEarlier || !primedMatchedBefore || !primedMatchedAfter))
      return failure();
    if (primedLater) {
      *primedMatchedBefore =
          matcher.find_nearest(primedLater, [this](Operation *candidate) {
            return candidate == primedEarlier;
          });
    }
    matchedBefore = matcher.find_nearest(later, reachesEarlier);

    auto parameterConstant =
        operation.getParameters().front().getDefiningOp<arith::ConstantOp>();
    if (change == ParameterChange::InPlaceMutation) {
      rewriter.modifyOpInPlace(parameterConstant, [&] {
        parameterConstant->setAttr("value", rewriter.getF64FloatAttr(2.0));
      });
    } else {
      rewriter.setInsertionPoint(parameterConstant);
      auto replacement = arith::ConstantFloatOp::create(
          rewriter, operation.getLoc(), rewriter.getF64Type(),
          llvm::APFloat(2.0));
      rewriter.replaceOp(parameterConstant, replacement.getResult());
    }

    matchedAfter = matcher.find_nearest(later, reachesEarlier);
    if (primedLater)
      *primedMatchedAfter =
          matcher.find_nearest(primedLater, [this](Operation *candidate) {
            return candidate == primedEarlier;
          });
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  bool &matchedBefore;
  bool &matchedAfter;
  ParameterChange change;
  Operation *primedLater;
  Operation *primedEarlier;
  bool *primedMatchedBefore;
  bool *primedMatchedAfter;
};

class CancelHadamardThenModifyUser
    : public OpRewritePattern<cudaq::quake::HOp> {
public:
  CancelHadamardThenModifyUser(
      MLIRContext *context, cudaq::opt::CommutationAwareRewriteMatcher &matcher,
      bool &matchedAfter)
      : OpRewritePattern(context), matcher(matcher),
        matchedAfter(matchedAfter) {}

  LogicalResult matchAndRewrite(cudaq::quake::HOp later,
                                PatternRewriter &rewriter) const override {
    if (later->getNumResults() != 2)
      return failure();
    Operation *earlier = matcher.find_nearest(later, [&](Operation *candidate) {
      return isa<cudaq::quake::HOp>(candidate) &&
             matcher.have_same_ordered_quantum_operands(later, candidate);
    });
    if (!earlier || !later->getResult(0).hasOneUse() ||
        !later->getResult(1).hasOneUse())
      return failure();

    Operation *user = *later->getResult(0).getUsers().begin();
    if (user != *later->getResult(1).getUsers().begin() ||
        !isa<cudaq::quake::XOp>(user) || user->getNumResults() != 2 ||
        !user->getResult(0).hasOneUse())
      return failure();
    Operation *next = *user->getResult(0).getUsers().begin();
    if (!isa<cudaq::quake::XOp>(next))
      return failure();

    rewriter.replaceOp(later, earlier->getOperands());
    rewriter.eraseOp(earlier);
    rewriter.modifyOpInPlace(user, [&] {
      user->setAttr("unsupported_mutation", rewriter.getUnitAttr());
    });
    matchedAfter = matcher.find_nearest(
        next, [user](Operation *candidate) { return candidate == user; });
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  bool &matchedAfter;
};

class InsertOpaqueClassicalCallAfterAnalysis
    : public OpRewritePattern<cudaq::quake::XOp> {
public:
  InsertOpaqueClassicalCallAfterAnalysis(
      MLIRContext *context, cudaq::opt::CommutationAwareRewriteMatcher &matcher,
      bool &inserted, bool &matchedBefore, bool &matchedAfter)
      : OpRewritePattern(context), matcher(matcher), inserted(inserted),
        matchedBefore(matchedBefore), matchedAfter(matchedAfter) {}

  LogicalResult matchAndRewrite(cudaq::quake::XOp later,
                                PatternRewriter &rewriter) const override {
    auto earlier =
        later.getTargets().front().getDefiningOp<cudaq::quake::XOp>();
    if (inserted || !earlier || earlier.getControls().size() != 1)
      return failure();

    auto targetUnwrap =
        earlier.getTargets().front().getDefiningOp<cudaq::quake::UnwrapOp>();
    auto allocation =
        targetUnwrap
            ? targetUnwrap.getRefValue().getDefiningOp<cudaq::quake::AllocaOp>()
            : cudaq::quake::AllocaOp{};
    if (!allocation)
      return failure();

    auto reachesEarlier = [earlier](Operation *candidate) {
      return candidate == earlier;
    };
    matchedBefore = matcher.find_nearest(later, reachesEarlier);
    if (!matchedBefore)
      return failure();

    inserted = true;
    rewriter.setInsertionPointAfter(allocation);
    func::CallOp::create(rewriter, later.getLoc(), "opaque_classical",
                         TypeRange{}, ValueRange{});
    matchedAfter = matcher.find_nearest(later, reachesEarlier);
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  bool &inserted;
  bool &matchedBefore;
  bool &matchedAfter;
};

class MoveClassicalOperationAfterAnalysis
    : public OpRewritePattern<cudaq::quake::XOp> {
public:
  MoveClassicalOperationAfterAnalysis(
      MLIRContext *context, cudaq::opt::CommutationAwareRewriteMatcher &matcher,
      Operation *sourceProbe, Operation *destinationProbe, Operation *toMove,
      bool &moved, bool &sourceValidAfter, bool &destinationValidAfter)
      : OpRewritePattern(context), matcher(matcher), sourceProbe(sourceProbe),
        destinationProbe(destinationProbe), toMove(toMove), moved(moved),
        sourceValidAfter(sourceValidAfter),
        destinationValidAfter(destinationValidAfter) {}

  LogicalResult matchAndRewrite(cudaq::quake::XOp operation,
                                PatternRewriter &rewriter) const override {
    if (moved || operation != destinationProbe ||
        !matcher.has_distinct_quantum_operands(sourceProbe) ||
        !matcher.has_distinct_quantum_operands(destinationProbe))
      return failure();

    rewriter.moveOpBefore(toMove, operation);
    moved = true;
    sourceValidAfter = matcher.has_distinct_quantum_operands(sourceProbe);
    destinationValidAfter =
        matcher.has_distinct_quantum_operands(destinationProbe);
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  Operation *sourceProbe;
  Operation *destinationProbe;
  Operation *toMove;
  bool &moved;
  bool &sourceValidAfter;
  bool &destinationValidAfter;
};

struct EraseListener : public RewriterBase::Listener {
  void notifyOperationErased(Operation *) override { ++eraseCount; }
  unsigned eraseCount = 0;
};

TEST_F(CommutationAwareRewriteTest,
       FindsNearestWithoutOpOrderAndUsesAdjacentFastPath) {
  auto module = parseModule(R"mlir(
    module {
      func.func @nearest() {
        %q = quake.null_wire
        %h0 = quake.h %q : (!quake.wire) -> !quake.wire
        %h1 = quake.h %h0 : (!quake.wire) -> !quake.wire
        %h2 = quake.h %h1 : (!quake.wire) -> !quake.wire
        quake.sink %h2 : !quake.wire
        return
      }
      func.func @adjacent_cancel() {
        %reference = quake.alloca !quake.ref
        %wire = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        %h0 = quake.h %wire : (!quake.wire) -> !quake.wire
        %h1 = quake.h %h0 : (!quake.wire) -> !quake.wire
        quake.wrap %h1 to %reference : !quake.wire, !quake.ref
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto function = getFunction(*module, "nearest");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 3u);
  Block &block = function.getBody().front();
  block.invalidateOpOrder();
  ASSERT_FALSE(block.isOpOrderValid());
  Operation *match = driver.get_matcher().find_nearest(
      operators[2], [&](Operation *candidate) {
        return isa<cudaq::quake::HOp>(candidate) &&
               driver.get_matcher().have_same_ordered_quantum_operands(
                   operators[2], candidate);
      });

  EXPECT_EQ(match, operators[1]);
  EXPECT_FALSE(block.isOpOrderValid());
  auto cancelFunction = getFunction(*module, "adjacent_cancel");
  driver.get_patterns().add<CancelHadamard>(&context, driver.get_matcher());
  EXPECT_TRUE(succeeded(driver.run(cancelFunction.getBody())));
  EXPECT_TRUE(cancelFunction.getOps<cudaq::quake::HOp>().empty());
  EXPECT_EQ(driver.get_statistics().analysisBuilds, 0u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(CommutationAwareRewriteTest,
       ForwardsConfiguredListenerRejectsReuseAndInvalidatesChangedRelations) {
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
      func.func @same_owner_then_unsupported_mutation() {
        %control = quake.null_wire
        %target = quake.null_wire
        %h0:2 = quake.h [%control] %target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %h1:2 = quake.h [%h0#0] %h0#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %x0:2 = quake.x [%h1#0] %h1#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %x1:2 = quake.x [%x0#0] %x0#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %x1#0 : !quake.wire
        quake.sink %x1#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "cancel");

  EraseListener listener;
  GreedyRewriteConfig config;
  config.setListener(&listener);
  cudaq::opt::CommutationAwareRewriteDriver driver(context, config);
  driver.get_patterns().add<CancelHadamard>(&context, driver.get_matcher());

  EXPECT_TRUE(succeeded(driver.run(function.getBody())));
  EXPECT_TRUE(function.getOps<cudaq::quake::HOp>().empty());
  EXPECT_GT(listener.eraseCount, 0u);
  auto statistics = driver.get_statistics();
  EXPECT_EQ(statistics.analysisBuilds, 1u);
  EXPECT_EQ(statistics.fallbackRebuilds, 0u);
  EXPECT_TRUE(failed(driver.run(function.getBody())));

  // Replacing both later H results rewires two operands owned by the same X.
  // Both expected modification callbacks must be consumed so the subsequent
  // unsupported in-place mutation still discards and rebuilds live analysis.
  auto mutationFunction =
      getFunction(*module, "same_owner_then_unsupported_mutation");
  cudaq::opt::CommutationAwareRewriteDriver mutationDriver(context);
  bool matchedAfterMutation = false;
  mutationDriver.get_patterns().add<CancelHadamardThenModifyUser>(
      &context, mutationDriver.get_matcher(), matchedAfterMutation);
  EXPECT_TRUE(succeeded(mutationDriver.run(mutationFunction.getBody())));
  EXPECT_TRUE(matchedAfterMutation);
  EXPECT_TRUE(mutationFunction.getOps<cudaq::quake::HOp>().empty());
  auto mutationStatistics = mutationDriver.get_statistics();
  EXPECT_EQ(mutationStatistics.analysisBuilds, 2u);
  EXPECT_EQ(mutationStatistics.fallbackRebuilds, 1u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(CommutationAwareRewriteTest,
       InvalidatesAnalysisForInsertedOpaqueClassicalCall) {
  auto module = parseModule(R"mlir(
    module {
      func.func private @opaque_classical()
      func.func @insert_call_boundary() {
        %control = quake.alloca !quake.ref
        %target = quake.alloca !quake.ref
        %controlWire = quake.unwrap %control : (!quake.ref) -> !quake.wire
        %targetWire = quake.unwrap %target : (!quake.ref) -> !quake.wire
        %x0:2 = quake.x [%controlWire] %targetWire
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %x1:2 = quake.x [%x0#0] %x0#1
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.wrap %x1#0 to %control : !quake.wire, !quake.ref
        quake.wrap %x1#1 to %target : !quake.wire, !quake.ref
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  auto function = getFunction(*module, "insert_call_boundary");
  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  bool inserted = false;
  bool matchedBefore = false;
  bool matchedAfter = true;
  driver.get_patterns().add<InsertOpaqueClassicalCallAfterAnalysis>(
      &context, driver.get_matcher(), inserted, matchedBefore, matchedAfter);

  EXPECT_TRUE(succeeded(driver.run(function.getBody())));
  EXPECT_TRUE(matchedBefore);
  EXPECT_FALSE(matchedAfter);
  auto statistics = driver.get_statistics();
  EXPECT_EQ(statistics.analysisBuilds, 2u);
  EXPECT_EQ(statistics.fallbackRebuilds, 1u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(CommutationAwareRewriteTest, RebuildsBothBlocksAfterOperationMove) {
  auto module = parseModule(R"mlir(
    module {
      func.func @move_between_blocks() {
        %unused = arith.constant 0 : i64
        %outerControl = quake.null_wire
        %outerTarget = quake.null_wire
        %outer:2 = quake.x [%outerControl] %outerTarget
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %outer#0 : !quake.wire
        quake.sink %outer#1 : !quake.wire
        cc.scope {
          %innerControl = quake.null_wire
          %innerTarget = quake.null_wire
          %inner:2 = quake.x [%innerControl] %innerTarget
              : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
          quake.sink %inner#0 : !quake.wire
          quake.sink %inner#1 : !quake.wire
          cc.continue
        }
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  auto function = getFunction(*module, "move_between_blocks");
  auto sourceOperators = getOperators(function);
  ASSERT_EQ(sourceOperators.size(), 1u);
  auto scope = *function.getBody().front().getOps<cudaq::cc::ScopeOp>().begin();
  auto destinationOperators = llvm::to_vector(
      scope.getInitRegion().front().getOps<cudaq::quake::XOp>());
  ASSERT_EQ(destinationOperators.size(), 1u);
  auto constants =
      llvm::to_vector(function.getBody().front().getOps<arith::ConstantOp>());
  ASSERT_EQ(constants.size(), 1u);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  bool moved = false;
  bool sourceValidAfter = false;
  bool destinationValidAfter = false;
  driver.get_patterns().add<MoveClassicalOperationAfterAnalysis>(
      &context, driver.get_matcher(), sourceOperators[0],
      destinationOperators[0], constants[0], moved, sourceValidAfter,
      destinationValidAfter);

  EXPECT_TRUE(succeeded(driver.run(function.getBody())));
  EXPECT_TRUE(sourceValidAfter);
  EXPECT_TRUE(destinationValidAfter);
  auto statistics = driver.get_statistics();
  EXPECT_EQ(statistics.analysisBuilds, 4u);
  EXPECT_EQ(statistics.fallbackRebuilds, 2u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(CommutationAwareRewriteTest,
       MaintainsNestedRelationsAfterClassicalChanges) {
  auto module = parseModule(R"mlir(
    module {
      func.func @nested_classical_replacement() {
        %theta0 = arith.constant 1.0 : f64
        %theta1 = arith.constant 1.0 : f64
        %zero = arith.constant 0.0 : f64
        %outerControl = quake.null_wire
        %outerTarget = quake.null_wire
        %outerH0:2 = quake.h [%outerControl] %outerTarget : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %outerH1:2 = quake.h [%outerH0#0] %outerH0#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %outerH1#0 : !quake.wire
        quake.sink %outerH1#1 : !quake.wire
        cc.scope {
          %q = quake.null_wire
          %u0 = quake.u3 (%theta0, %zero, %zero) %q : (f64, f64, f64, !quake.wire) -> !quake.wire
          %u1 = quake.u3 (%theta1, %zero, %zero) %u0 : (f64, f64, f64, !quake.wire) -> !quake.wire
          %u2 = quake.u3 (%theta0, %zero, %zero) %u1 : (f64, f64, f64, !quake.wire) -> !quake.wire
          %h = quake.h %u2 : (!quake.wire) -> !quake.wire
          quake.sink %h : !quake.wire
          cc.continue
        }
        return
      }
      func.func @nested_captured_parameter_mutation() {
        %theta0 = arith.constant 1.0 : f64
        %theta1 = arith.constant 1.0 : f64
        %zero = arith.constant 0.0 : f64
        cc.scope {
          %q = quake.null_wire
          %u0 = quake.u3 (%theta0, %zero, %zero) %q : (f64, f64, f64, !quake.wire) -> !quake.wire
          %u1 = quake.u3 (%theta1, %zero, %zero) %u0 : (f64, f64, f64, !quake.wire) -> !quake.wire
          %u2 = quake.u3 (%theta0, %zero, %zero) %u1 : (f64, f64, f64, !quake.wire) -> !quake.wire
          %h = quake.h %u2 : (!quake.wire) -> !quake.wire
          quake.sink %h : !quake.wire
          cc.continue
        }
        cc.scope {
          %control = quake.null_wire
          %target = quake.null_wire
          %h0:2 = quake.h [%control] %target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
          %h1:2 = quake.h [%h0#0] %h0#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
          quake.sink %h1#0 : !quake.wire
          quake.sink %h1#1 : !quake.wire
          cc.continue
        }
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  auto function = getFunction(*module, "nested_classical_replacement");
  auto outerOperators = getOperators(function);
  ASSERT_EQ(outerOperators.size(), 2u);
  GreedyRewriteConfig config;
  config.enableFolding(false).enableConstantCSE(false);
  cudaq::opt::CommutationAwareRewriteDriver driver(context, config);
  bool outerMatchedBefore = false;
  bool outerMatchedAfter = false;
  bool matchedBefore = false;
  bool matchedAfter = true;
  driver.get_patterns().add<ChangeOperatorParameter>(
      &context, driver.get_matcher(), matchedBefore, matchedAfter,
      ParameterChange::Replacement, outerOperators[1], outerOperators[0],
      &outerMatchedBefore, &outerMatchedAfter);

  EXPECT_TRUE(succeeded(driver.run(function.getBody())));
  EXPECT_TRUE(outerMatchedBefore);
  EXPECT_TRUE(outerMatchedAfter);
  EXPECT_TRUE(matchedBefore);
  EXPECT_FALSE(matchedAfter);
  auto statistics = driver.get_statistics();
  EXPECT_EQ(statistics.analysisBuilds, 2u);
  EXPECT_EQ(statistics.fallbackRebuilds, 0u);

  // A nested analysis can depend on a classical value captured from its
  // parent block. An unexplained in-place change has no replacement use-list
  // from which the listener could recover those dependencies.
  auto mutationFunction =
      getFunction(*module, "nested_captured_parameter_mutation");
  GreedyRewriteConfig mutationConfig;
  mutationConfig.enableFolding(false).enableConstantCSE(false);
  cudaq::opt::CommutationAwareRewriteDriver mutationDriver(context,
                                                           mutationConfig);
  bool mutationMatchedBefore = false;
  bool mutationMatchedAfter = true;
  auto mutationScopes = llvm::to_vector(
      mutationFunction.getBody().front().getOps<cudaq::cc::ScopeOp>());
  ASSERT_EQ(mutationScopes.size(), 2u);
  auto siblingOperators = llvm::to_vector(
      mutationScopes[1].getInitRegion().front().getOps<cudaq::quake::HOp>());
  ASSERT_EQ(siblingOperators.size(), 2u);
  bool siblingMatchedBefore = false;
  bool siblingMatchedAfter = false;
  mutationDriver.get_patterns().add<ChangeOperatorParameter>(
      &context, mutationDriver.get_matcher(), mutationMatchedBefore,
      mutationMatchedAfter, ParameterChange::InPlaceMutation,
      siblingOperators[1], siblingOperators[0], &siblingMatchedBefore,
      &siblingMatchedAfter);
  EXPECT_TRUE(succeeded(mutationDriver.run(mutationFunction.getBody())));
  EXPECT_TRUE(mutationMatchedBefore);
  EXPECT_FALSE(mutationMatchedAfter);
  EXPECT_TRUE(siblingMatchedBefore);
  EXPECT_TRUE(siblingMatchedAfter);
  auto mutationStatistics = mutationDriver.get_statistics();
  EXPECT_EQ(mutationStatistics.analysisBuilds, 3u);
  EXPECT_EQ(mutationStatistics.fallbackRebuilds, 1u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
