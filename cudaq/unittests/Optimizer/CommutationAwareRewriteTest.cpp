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
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <gtest/gtest.h>
#include <iterator>

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

template <typename QOP>
class CancelSelfInverse : public OpRewritePattern<QOP> {
public:
  CancelSelfInverse(MLIRContext *context,
                    cudaq::opt::CommutationAwareRewriteMatcher &matcher)
      : OpRewritePattern<QOP>(context), matcher(matcher) {}

  LogicalResult matchAndRewrite(QOP later,
                                PatternRewriter &rewriter) const override {
    Operation *earlier = matcher.findNearest(later, [&](Operation *candidate) {
      return isa<QOP>(candidate) &&
             matcher.haveSameOrderedQuantumOperands(later, candidate);
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

using CancelHadamard = CancelSelfInverse<cudaq::quake::HOp>;
using CancelX = CancelSelfInverse<cudaq::quake::XOp>;

class ReplaceOperatorParameter : public OpRewritePattern<cudaq::quake::U3Op> {
public:
  ReplaceOperatorParameter(MLIRContext *context,
                           cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                           bool &matchedBefore, bool &matchedAfter,
                           Operation *primedLater = nullptr,
                           Operation *primedEarlier = nullptr,
                           bool *primedMatched = nullptr)
      : OpRewritePattern(context), matcher(matcher),
        matchedBefore(matchedBefore), matchedAfter(matchedAfter),
        primedLater(primedLater), primedEarlier(primedEarlier),
        primedMatched(primedMatched) {}

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
    if (primedLater && (!primedEarlier || !primedMatched))
      return failure();
    if (primedLater) {
      *primedMatched =
          matcher.findNearest(primedLater, [this](Operation *candidate) {
            return candidate == primedEarlier;
          });
    }
    matchedBefore = matcher.findNearest(later, reachesEarlier);

    auto replacedConstant =
        operation.getParameters().front().getDefiningOp<arith::ConstantOp>();
    rewriter.setInsertionPoint(replacedConstant);
    auto replacement = arith::ConstantFloatOp::create(
        rewriter, operation.getLoc(), rewriter.getF64Type(),
        llvm::APFloat(2.0));
    rewriter.replaceOp(replacedConstant, replacement.getResult());

    matchedAfter = matcher.findNearest(later, reachesEarlier);
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  bool &matchedBefore;
  bool &matchedAfter;
  Operation *primedLater;
  Operation *primedEarlier;
  bool *primedMatched;
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
    Operation *earlier = matcher.findNearest(later, [&](Operation *candidate) {
      return isa<cudaq::quake::HOp>(candidate) &&
             matcher.haveSameOrderedQuantumOperands(later, candidate);
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
    matchedAfter = matcher.findNearest(
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
    matchedBefore = matcher.findNearest(later, reachesEarlier);
    if (!matchedBefore)
      return failure();

    inserted = true;
    rewriter.setInsertionPointAfter(allocation);
    func::CallOp::create(rewriter, later.getLoc(), "opaque_classical",
                         TypeRange{}, ValueRange{});
    matchedAfter = matcher.findNearest(later, reachesEarlier);
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
        !matcher.hasDistinctQuantumOperands(sourceProbe) ||
        !matcher.hasDistinctQuantumOperands(destinationProbe))
      return failure();

    rewriter.moveOpBefore(toMove, operation);
    moved = true;
    sourceValidAfter = matcher.hasDistinctQuantumOperands(sourceProbe);
    destinationValidAfter =
        matcher.hasDistinctQuantumOperands(destinationProbe);
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

TEST_F(CommutationAwareRewriteTest, FindsNearestEarlierEndpoint) {
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
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto function = getFunction(*module, "nearest");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 3u);
  Block &block = function.getBody().front();
  block.invalidateOpOrder();
  ASSERT_FALSE(block.isOpOrderValid());
  Operation *match =
      driver.getMatcher().findNearest(operators[2], [&](Operation *candidate) {
        return isa<cudaq::quake::HOp>(candidate) &&
               driver.getMatcher().haveSameOrderedQuantumOperands(operators[2],
                                                                  candidate);
      });

  EXPECT_EQ(match, operators[1]);
  EXPECT_FALSE(block.isOpOrderValid());
  EXPECT_TRUE(driver.getMatcher().hasDistinctQuantumOperands(operators[2]));
  EXPECT_EQ(driver.getStatistics().analysisBuilds, 0u);
}

TEST_F(CommutationAwareRewriteTest,
       AcceptsDistinctDynamicPauliOperandsAndAdjacentEndpoint) {
  auto module = parseModule(R"mlir(
    module {
      func.func @dynamic_pauli(%word: !cc.charspan) {
        %angle = arith.constant 5.0e-1 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %earlier:2 = quake.exp_pauli (%angle) %q0, %q1 to %word
            : (f64, !quake.wire, !quake.wire, !cc.charspan)
              -> (!quake.wire, !quake.wire)
        %later:2 = quake.x [%earlier#0] %earlier#1
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %later#0 : !quake.wire
        quake.sink %later#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto operators = getOperators(getFunction(*module, "dynamic_pauli"));
  ASSERT_EQ(operators.size(), 2u);
  EXPECT_TRUE(driver.getMatcher().hasDistinctQuantumOperands(operators[0]));
  EXPECT_EQ(driver.getMatcher().findNearest(operators[1],
                                            [&](Operation *candidate) {
                                              return candidate == operators[0];
                                            }),
            operators[0]);
  EXPECT_EQ(driver.getStatistics().analysisBuilds, 1u);
}

TEST_F(CommutationAwareRewriteTest, RequiresCompleteMultiWireFrontier) {
  auto module = parseModule(R"mlir(
    module {
      func.func @complete_frontier() {
        %control = quake.null_wire
        %target = quake.null_wire
        %other = quake.null_wire
        %h0:2 = quake.h [%control] %target
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %z:2 = quake.z [%h0#0] %other
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %h1:2 = quake.h [%z#0] %h0#1
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %h1#0 : !quake.wire
        quake.sink %h1#1 : !quake.wire
        quake.sink %z#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto function = getFunction(*module, "complete_frontier");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 3u);
  Block &block = function.getBody().front();
  block.invalidateOpOrder();
  ASSERT_FALSE(block.isOpOrderValid());
  Operation *match =
      driver.getMatcher().findNearest(operators[2], [&](Operation *candidate) {
        return candidate == operators[0];
      });

  EXPECT_EQ(match, operators[0]);
  EXPECT_FALSE(block.isOpOrderValid());
}

TEST_F(CommutationAwareRewriteTest, StopsAtIncompleteMultiWireFrontier) {
  auto module = parseModule(R"mlir(
    module {
      func.func @incomplete_frontier() {
        %control = quake.null_wire
        %target = quake.null_wire
        %candidate = quake.x %target : (!quake.wire) -> !quake.wire
        %anchor:2 = quake.x [%control] %candidate
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %anchor#0 : !quake.wire
        quake.sink %anchor#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto operators = getOperators(getFunction(*module, "incomplete_frontier"));
  ASSERT_EQ(operators.size(), 2u);
  unsigned predicateCalls = 0;
  Operation *match =
      driver.getMatcher().findNearest(operators[1], [&](Operation *candidate) {
        ++predicateCalls;
        return candidate == operators[0];
      });

  EXPECT_FALSE(match);
  EXPECT_EQ(predicateCalls, 1u);
}

TEST_F(CommutationAwareRewriteTest, RejectsBranchedProducerResult) {
  auto module = parseModule(R"mlir(
    module {
      func.func @branched_result() {
        %q = quake.null_wire
        %h0 = quake.h %q : (!quake.wire) -> !quake.wire
        %h1 = quake.h %h0 : (!quake.wire) -> !quake.wire
        quake.sink %h1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto function = getFunction(*module, "branched_result");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 2u);

  OpBuilder builder(&context);
  builder.setInsertionPoint(operators[1]);
  cudaq::quake::SinkOp::create(builder, operators[0]->getLoc(), TypeRange{},
                               operators[0]->getResult(0));
  ASSERT_EQ(std::distance(operators[0]->getResult(0).use_begin(),
                          operators[0]->getResult(0).use_end()),
            2);
  EXPECT_FALSE(
      driver.getMatcher().findNearest(operators[1], [&](Operation *candidate) {
        return candidate == operators[0];
      }));
}

TEST_F(CommutationAwareRewriteTest, RejectsUnsupportedCallBoundary) {
  auto module = parseModule(R"mlir(
    module {
      func.func private @opaque(!quake.wire) -> !quake.wire
      func.func @call_boundary() {
        %q = quake.null_wire
        %h0 = quake.h %q : (!quake.wire) -> !quake.wire
        %called = func.call @opaque(%h0) : (!quake.wire) -> !quake.wire
        %h1 = quake.h %called : (!quake.wire) -> !quake.wire
        quake.sink %h1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  cudaq::opt::CommutationAwareRewriteDriver driver(context);
  auto operators = getOperators(getFunction(*module, "call_boundary"));
  ASSERT_EQ(operators.size(), 2u);
  unsigned predicateCalls = 0;
  EXPECT_FALSE(
      driver.getMatcher().findNearest(operators[1], [&](Operation *candidate) {
        ++predicateCalls;
        return candidate == operators[0];
      }));
  EXPECT_EQ(predicateCalls, 0u);
}

TEST_F(CommutationAwareRewriteTest, KeepsReplacementAtLaterAnchorLocation) {
  auto module = parseModule(R"mlir(
    module {
      func.func @later_location() {
        %q = quake.null_wire
        %s0 = quake.s %q : (!quake.wire) -> !quake.wire loc("earlier")
        %s1 = quake.s %s0 : (!quake.wire) -> !quake.wire loc("later")
        quake.sink %s1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  PassManager manager(&context);
  manager.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeSimplify());
  ASSERT_TRUE(succeeded(manager.run(*module)));

  auto function = getFunction(*module, "later_location");
  auto replacements = llvm::to_vector(function.getOps<cudaq::quake::ZOp>());
  ASSERT_EQ(replacements.size(), 1u);
  auto location = dyn_cast<NameLoc>(replacements.front().getLoc());
  ASSERT_TRUE(location);
  EXPECT_EQ(location.getName(), "later");
  EXPECT_TRUE(function.getOps<cudaq::quake::SOp>().empty());
  EXPECT_TRUE(succeeded(verify(*module)));
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
      func.func @repeated_wire() {
        %q = quake.null_wire
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
      driver.getMatcher().findNearest(operators[1], [&](Operation *candidate) {
        return candidate == operators[0];
      }));
  EXPECT_FALSE(driver.getMatcher().haveSameOrderedQuantumOperands(
      operators[0], operators[1]));
  EXPECT_FALSE(driver.getMatcher().hasDistinctQuantumOperands(operators[0]));
  EXPECT_FALSE(driver.getMatcher().hasDistinctQuantumOperands(operators[1]));
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
      driver.getMatcher().findNearest(operators[1], [&](Operation *candidate) {
        return candidate == operators[0];
      }));
  EXPECT_FALSE(driver.getMatcher().haveSameOrderedQuantumOperands(
      operators[0], operators[1]));
  EXPECT_FALSE(driver.getMatcher().hasDistinctQuantumOperands(operators[0]));
  EXPECT_FALSE(driver.getMatcher().hasDistinctQuantumOperands(operators[1]));
  EXPECT_EQ(driver.getStatistics().analysisBuilds, 1u);
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
      func.func @classical_replacement() {
        %theta0 = arith.constant 1.0 : f64
        %theta1 = arith.constant 1.0 : f64
        %zero = arith.constant 0.0 : f64
        %q = quake.null_wire
        %u0 = quake.u3 (%theta0, %zero, %zero) %q : (f64, f64, f64, !quake.wire) -> !quake.wire
        %u1 = quake.u3 (%theta1, %zero, %zero) %u0 : (f64, f64, f64, !quake.wire) -> !quake.wire
        %u2 = quake.u3 (%theta0, %zero, %zero) %u1 : (f64, f64, f64, !quake.wire) -> !quake.wire
        %h = quake.h %u2 : (!quake.wire) -> !quake.wire
        quake.sink %h : !quake.wire
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
  // same search must stop without rebuilding or returning that stale relation.
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
  EXPECT_EQ(parameterStatistics.analysisBuilds, 1u);
  EXPECT_EQ(parameterStatistics.fallbackRebuilds, 0u);

  // Replacing both later H results rewires two operands owned by the same X.
  // Both expected modification callbacks must be consumed so the subsequent
  // unsupported in-place mutation still discards and rebuilds live analysis.
  auto mutationFunction =
      getFunction(*module, "same_owner_then_unsupported_mutation");
  cudaq::opt::CommutationAwareRewriteDriver mutationDriver(context);
  bool matchedAfterMutation = false;
  mutationDriver.getPatterns().add<CancelHadamardThenModifyUser>(
      &context, mutationDriver.getMatcher(), matchedAfterMutation);
  EXPECT_TRUE(succeeded(mutationDriver.run(mutationFunction.getBody())));
  EXPECT_TRUE(matchedAfterMutation);
  EXPECT_TRUE(mutationFunction.getOps<cudaq::quake::HOp>().empty());
  auto mutationStatistics = mutationDriver.getStatistics();
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
  driver.getPatterns().add<InsertOpaqueClassicalCallAfterAnalysis>(
      &context, driver.getMatcher(), inserted, matchedBefore, matchedAfter);

  EXPECT_TRUE(succeeded(driver.run(function.getBody())));
  EXPECT_TRUE(inserted);
  EXPECT_TRUE(matchedBefore);
  EXPECT_FALSE(matchedAfter);
  auto statistics = driver.getStatistics();
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
  driver.getPatterns().add<MoveClassicalOperationAfterAnalysis>(
      &context, driver.getMatcher(), sourceOperators[0],
      destinationOperators[0], constants[0], moved, sourceValidAfter,
      destinationValidAfter);

  EXPECT_TRUE(succeeded(driver.run(function.getBody())));
  EXPECT_TRUE(moved);
  EXPECT_TRUE(sourceValidAfter);
  EXPECT_TRUE(destinationValidAfter);
  auto statistics = driver.getStatistics();
  EXPECT_EQ(statistics.analysisBuilds, 4u);
  EXPECT_EQ(statistics.fallbackRebuilds, 2u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(CommutationAwareRewriteTest,
       ClearsRelationsInAnalyzedBlocksUsingAReplacement) {
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
    })mlir");
  ASSERT_TRUE(module);

  auto function = getFunction(*module, "nested_classical_replacement");
  auto outerOperators = getOperators(function);
  ASSERT_EQ(outerOperators.size(), 2u);
  GreedyRewriteConfig config;
  config.enableFolding(false).enableConstantCSE(false);
  cudaq::opt::CommutationAwareRewriteDriver driver(context, config);
  bool outerMatched = false;
  bool matchedBefore = false;
  bool matchedAfter = true;
  driver.getPatterns().add<ReplaceOperatorParameter>(
      &context, driver.getMatcher(), matchedBefore, matchedAfter,
      outerOperators[1], outerOperators[0], &outerMatched);

  EXPECT_TRUE(succeeded(driver.run(function.getBody())));
  EXPECT_TRUE(outerMatched);
  EXPECT_TRUE(matchedBefore);
  EXPECT_FALSE(matchedAfter);
  auto statistics = driver.getStatistics();
  EXPECT_EQ(statistics.analysisBuilds, 2u);
  EXPECT_EQ(statistics.fallbackRebuilds, 0u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

} // namespace
