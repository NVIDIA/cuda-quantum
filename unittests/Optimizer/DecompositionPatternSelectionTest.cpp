/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Include the implementation file that we are testing
#include "DecompositionPatternSelection.cpp"
#include "DecompositionPatterns.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace mlir;

namespace {
/// A mock pattern class
class PatternTest : public mlir::RewritePattern {
public:
  PatternTest(llvm::StringRef patternName, MLIRContext *context)
      : mlir::RewritePattern(patternName, 0, context, {}) {
    setDebugName(patternName);
  }
};

/// A mock pattern type for testing.
class PatternTypeTest : public cudaq::DecompositionPatternType {
public:
  PatternTypeTest(llvm::StringRef patternName, llvm::StringRef sourceOp,
                  std::vector<llvm::StringRef> targetOps)
      : patternName(patternName), sourceOp(sourceOp), targetOps(targetOps) {}

  llvm::StringRef getSourceOp() const override { return sourceOp; }

  llvm::ArrayRef<llvm::StringRef> getTargetOps() const override {
    return targetOps;
  }

  llvm::StringRef getPatternName() const override { return patternName; }

  std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const override {
    return std::make_unique<PatternTest>(patternName, context);
  };

private:
  llvm::StringRef patternName;
  llvm::StringRef sourceOp;
  std::vector<llvm::StringRef> targetOps;
};

/// Create a test decomposition graph with the following patterns. The arrow
// "->" should be read as "decomposes to".
/// x -> x(1) -> x(2) -> x(3)
/// y -> y(1) -> y(2) -> y(3)
/// z -> z(1)+x(1)
/// z(1) -> z(2)+x(2)
/// z(2) -> z(3)+x(3)
/// z -> h -> z(1)
DecompositionGraph createTestGraph() {
  // Decompose x -> x(1) -> x(2) -> x(3)
  auto pattern_x1 = std::make_unique<PatternTypeTest>(
      "pattern_x1", "x", std::vector<llvm::StringRef>{"x(1)"});
  auto pattern_x2 = std::make_unique<PatternTypeTest>(
      "pattern_x2", "x(1)", std::vector<llvm::StringRef>{"x(2)"});
  auto pattern_x3 = std::make_unique<PatternTypeTest>(
      "pattern_x3", "x(2)", std::vector<llvm::StringRef>{"x(3)"});

  // Decompose y -> y(1) -> y(2) -> y(3)
  auto pattern_y1 = std::make_unique<PatternTypeTest>(
      "pattern_y1", "y", std::vector<llvm::StringRef>{"y(1)"});
  auto pattern_y2 = std::make_unique<PatternTypeTest>(
      "pattern_y2", "y(1)", std::vector<llvm::StringRef>{"y(2)"});
  auto pattern_y3 = std::make_unique<PatternTypeTest>(
      "pattern_y3", "y(2)", std::vector<llvm::StringRef>{"y(3)"});

  // Decompose z similarly to x and y, but it creates "side effects" in the form
  // of extra x gates.
  auto pattern_z1 = std::make_unique<PatternTypeTest>(
      "pattern_z1", "z", std::vector<llvm::StringRef>{"z(1)", "x(1)"});
  auto pattern_z2 = std::make_unique<PatternTypeTest>(
      "pattern_z2", "z(1)", std::vector<llvm::StringRef>{"z(2)", "x(2)"});
  auto pattern_z3 = std::make_unique<PatternTypeTest>(
      "pattern_z3", "z(2)", std::vector<llvm::StringRef>{"z(3)", "x(3)"});

  // Another way to decompose z -> z(1), is side-effect free, but requires an
  // extra pattern.
  // z -> h -> z(1)
  auto pattern_zh1 = std::make_unique<PatternTypeTest>(
      "pattern_zh1", "z", std::vector<llvm::StringRef>{"h"});
  auto pattern_zh2 = std::make_unique<PatternTypeTest>(
      "pattern_zh2", "h", std::vector<llvm::StringRef>{"z(1)"});

  llvm::StringMap<std::unique_ptr<cudaq::DecompositionPatternType>> patterns;
  patterns.insert({pattern_x1->getPatternName(), std::move(pattern_x1)});
  patterns.insert({pattern_x2->getPatternName(), std::move(pattern_x2)});
  patterns.insert({pattern_x3->getPatternName(), std::move(pattern_x3)});
  patterns.insert({pattern_y1->getPatternName(), std::move(pattern_y1)});
  patterns.insert({pattern_y2->getPatternName(), std::move(pattern_y2)});
  patterns.insert({pattern_y3->getPatternName(), std::move(pattern_y3)});
  patterns.insert({pattern_z1->getPatternName(), std::move(pattern_z1)});
  patterns.insert({pattern_z2->getPatternName(), std::move(pattern_z2)});
  patterns.insert({pattern_z3->getPatternName(), std::move(pattern_z3)});
  patterns.insert({pattern_zh1->getPatternName(), std::move(pattern_zh1)});
  patterns.insert({pattern_zh2->getPatternName(), std::move(pattern_zh2)});
  return DecompositionGraph(std::move(patterns));
}

class BaseDecompositionPatternSelectionTest : public ::testing::Test {
protected:
  void SetUp() override {
    context = std::make_unique<MLIRContext>();
    context->loadDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                         func::FuncDialect, quake::QuakeDialect>();
    // set up graph in children classes
  }

  /// Whether an operation of type Op with nCtrls control qubits is legal on
  /// the target.
  template <typename Op>
  bool isLegal(const std::unique_ptr<ConversionTarget> &target,
               unsigned nCtrls = 0) {
    // Create a module with a single operation of type Op
    auto loc = UnknownLoc::get(context.get());
    auto module = ModuleOp::create(loc);
    OpBuilder builder(module.getBodyRegion());

    // Create a function to hold the operation
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_func", funcType);
    auto *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create n_qubits qubit wires
    SmallVector<Value> controls;
    auto wireType = quake::WireType::get(context.get());
    for (unsigned i = 0; i < nCtrls; ++i) {
      auto qubit = builder.create<quake::AllocaOp>(loc, wireType);
      controls.push_back(qubit.getResult());
    }
    auto targetQubit = builder.create<quake::AllocaOp>(loc, wireType);
    SmallVector<Value> targets{targetQubit};

    // Create the operation of type Op with the qubits
    auto op = builder.create<Op>(loc, controls, targets);

    // Get the operation pointer and check if it is legal
    Operation *operation_ptr = op.getOperation();
    return target->isLegal(operation_ptr).has_value();
  }

  /// Run `selectPatterns` on the current decomposition graph and return the
  /// selected patterns as a vector of sorted pattern names.
  std::vector<std::string>
  selectPatterns(const std::vector<std::string> &targetBasis,
                 const std::unordered_set<std::string> &disabledPatterns = {}) {
    auto convertToOperatorInfoSet =
        [](const std::vector<std::string> &targetBasis) {
          std::unordered_set<OperatorInfo> operatorInfoSet;
          for (const auto &target : targetBasis) {
            operatorInfoSet.insert(OperatorInfo(target));
          }
          return operatorInfoSet;
        };

    RewritePatternSet patterns(context.get());
    graph.selectPatterns(patterns, convertToOperatorInfoSet(targetBasis),
                         disabledPatterns);

    std::vector<std::string> selectedPatterns;
    for (const auto &pattern : patterns.getNativePatterns()) {
      selectedPatterns.push_back(pattern->getDebugName().str());
    }
    std::sort(selectedPatterns.begin(), selectedPatterns.end());
    return selectedPatterns;
  }

  std::unique_ptr<MLIRContext> context;
  DecompositionGraph graph;
};

/// Run pattern selection tests on a dummy graph.
class DummyDecompositionPatternSelectionTest
    : public BaseDecompositionPatternSelectionTest {
protected:
  void SetUp() override {
    BaseDecompositionPatternSelectionTest::SetUp();
    graph = createTestGraph();
  }
};

/// Run pattern selection tests on the full decomposition graph.
class FullDecompositionPatternSelectionTest
    : public BaseDecompositionPatternSelectionTest {
protected:
  void SetUp() override {
    BaseDecompositionPatternSelectionTest::SetUp();
    graph = DecompositionGraph::fromRegistry();
  }
};

//===----------------------------------------------------------------------===//
// Test BasisTarget
//===----------------------------------------------------------------------===//

TEST_F(BaseDecompositionPatternSelectionTest, BasisTargetParsesSimpleGates) {
  std::vector<std::string> basis{"h", "t", "x"};
  auto target = cudaq::createBasisTarget(*context, basis);
  EXPECT_TRUE(isLegal<quake::HOp>(target));
  EXPECT_TRUE(isLegal<quake::TOp>(target));
  EXPECT_TRUE(isLegal<quake::XOp>(target));

  EXPECT_FALSE(isLegal<quake::HOp>(target, 1));
  EXPECT_FALSE(isLegal<quake::TOp>(target, 1));
  EXPECT_FALSE(isLegal<quake::XOp>(target, 1));
  EXPECT_FALSE(isLegal<quake::ZOp>(target));
}

TEST_F(BaseDecompositionPatternSelectionTest,
       BasisTargetParsesControlledGates) {
  std::vector<std::string> basis{"x(1)", "z(2)"};
  auto target = cudaq::createBasisTarget(*context, basis);
  EXPECT_TRUE(isLegal<quake::XOp>(target, 1));
  EXPECT_TRUE(isLegal<quake::ZOp>(target, 2));

  EXPECT_FALSE(isLegal<quake::XOp>(target));
  EXPECT_FALSE(isLegal<quake::XOp>(target, 2));
  EXPECT_FALSE(isLegal<quake::ZOp>(target));
  EXPECT_FALSE(isLegal<quake::ZOp>(target, 1));
  EXPECT_FALSE(isLegal<quake::ZOp>(target, 3));
}

TEST_F(BaseDecompositionPatternSelectionTest,
       BasisTargetParsesArbitraryControls) {
  std::vector<std::string> basis{"x(n)"};
  auto target = cudaq::createBasisTarget(*context, basis);

  EXPECT_TRUE(isLegal<quake::XOp>(target, 0));
  EXPECT_TRUE(isLegal<quake::XOp>(target, 1));
  EXPECT_TRUE(isLegal<quake::XOp>(target, 2));
  EXPECT_TRUE(isLegal<quake::XOp>(target, 10));
}

//===----------------------------------------------------------------------===//
// Test selectDecompositionPatterns on dummy graph
//===----------------------------------------------------------------------===//

// Reminder: here are the fictional decompositions that we allow:
// y -> y(1) -> y(2) -> y(3)
// z -> z(1)+x(1)
// z(1) -> z(2)+x(2)
// z(2) -> z(3)+x(3)
// z -> h -> z(1)

TEST_F(DummyDecompositionPatternSelectionTest, SelectXPatterns) {
  std::vector<std::string> targetBasis{"x(3)"};
  auto selectedPatterns = selectPatterns(targetBasis);

  // gates x, x(1) and x(2) can be decomposed to x(3), using the pattern_x*
  // decomposition patterns:
  // - pattern_x1: decompose x into x(1)
  // - pattern_x2: decompose x(1) into x(2)
  // - pattern_x3: decompose x(2) into x(3)
  std::vector<std::string> exp{"pattern_x1", "pattern_x2", "pattern_x3"};
  EXPECT_EQ(selectedPatterns, exp);
}

TEST_F(DummyDecompositionPatternSelectionTest, SelectYPatterns) {
  std::vector<std::string> targetBasis{"y(2)"};
  auto selectedPatterns = selectPatterns(targetBasis);

  // gates y, y(1) can be decomposed to y(2), using the pattern_y*
  // decomposition patterns:
  // - pattern_y1: decompose y into y(1)
  // - pattern_y2: decompose y(1) into y(2)
  // pattern_y3 cannot be used, as it decomposes to y(3)
  std::vector<std::string> exp{"pattern_y1", "pattern_y2"};
  EXPECT_EQ(selectedPatterns, exp);
}

TEST_F(DummyDecompositionPatternSelectionTest, SelectZOverXPatterns) {
  std::vector<std::string> targetBasis{"z(2)", "x(3)"};
  auto selectedPatterns = selectPatterns(targetBasis);

  // The decomposition patterns for z also introduce x gates. As we allow both
  // x and z in the target basis, we can use the following z decomposition
  // patterns:
  // - pattern_x1: decompose x into x(1)
  // - pattern_x2: decompose x(1) into x(2)
  // - pattern_x3: decompose x(2) into x(3)
  // - pattern_z1: decompose z into z(1)+x(1)
  // - pattern_z2: decompose z(1) into z(2)+x(2)
  // - pattern_zh2: decompose h into z(1)
  // Pattern pattern_zh1 cannot be used, as z is already decomposed by
  // pattern_z1.
  std::vector<std::string> exp{"pattern_x1", "pattern_x2", "pattern_x3",
                               "pattern_z1", "pattern_z2", "pattern_zh2"};
  EXPECT_EQ(selectedPatterns, exp);
}

TEST_F(DummyDecompositionPatternSelectionTest, SelectZOverHPatterns) {
  std::vector<std::string> targetBasis{"z(1)"};
  auto selectedPatterns = selectPatterns(targetBasis);

  // The decomposition patterns for z also introduce x gates, but we do not
  // accept x gates here. We can therefore only use the z-over-h decomposition
  // patterns:
  // - pattern_zh1: decompose z into h
  // - pattern_zh2: decompose h into z(1)
  std::vector<std::string> exp{"pattern_zh1", "pattern_zh2"};
  EXPECT_EQ(selectedPatterns, exp);
}

TEST_F(DummyDecompositionPatternSelectionTest,
       SelectZOverHPatternsWithDisabledPatterns) {
  std::vector<std::string> targetBasis{"z(1)", "x(1)"};
  std::unordered_set<std::string> disabledPatterns{"pattern_z1"};
  auto selectedPatterns = selectPatterns(targetBasis, disabledPatterns);

  // If we only consider the target basis, then pattern_z1:
  // z -> z(1)+x(1)
  // would be selected. However, by disabling it we force the selection of the
  // pattern_zh1 instead.
  std::vector<std::string> exp{"pattern_x1", "pattern_zh1", "pattern_zh2"};
  EXPECT_EQ(selectedPatterns, exp);
}

//===----------------------------------------------------------------------===//
// Test selectDecompositionPatterns on the registered decomposition graph
//===----------------------------------------------------------------------===//

TEST_F(FullDecompositionPatternSelectionTest, DecomposeCCXToCZ) {
  std::vector<std::string> targetBasis{"h", "t", "z(1)"};
  auto selectedPatterns = selectPatterns(targetBasis);

  std::vector<std::string> exp{"CCXToCCZ", "CCZToCX", "CXToCZ", "SwapToCX"};
  EXPECT_EQ(selectedPatterns, exp);
}

} // namespace
