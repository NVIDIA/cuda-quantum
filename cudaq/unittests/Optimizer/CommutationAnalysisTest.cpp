/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;

using CommutationAnalysis = cudaq::quake::detail::CommutationAnalysis;
using commutation_reason = cudaq::quake::detail::commutation_reason;
using commutation_status = cudaq::quake::detail::commutation_status;

namespace {
class CommutationAnalysisTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<cf::ControlFlowDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<cudaq::cc::CCDialect>();
    context.loadDialect<cudaq::quake::QuakeDialect>();
  }

  OwningOpRef<ModuleOp> parseModule(llvm::StringRef source) {
    auto parsed = parseSourceString<ModuleOp>(source, &context);
    if (parsed && succeeded(verify(*parsed)))
      return parsed;
    return {};
  }

  static func::FuncOp getFunction(ModuleOp module, llvm::StringRef name) {
    auto function = module.lookupSymbol<func::FuncOp>(name);
    EXPECT_TRUE(function);
    return function;
  }

  static llvm::SmallVector<Operation *> getOperators(func::FuncOp function) {
    llvm::SmallVector<Operation *> operators;
    if (!function)
      return operators;
    for (Operation &operation : function.front())
      if (isa<cudaq::quake::OperatorInterface>(operation))
        operators.push_back(&operation);
    return operators;
  }

  // Check that both operand orders produce the expected detailed result.
  static void expectPair(CommutationAnalysis &analysis, Operation *lhs,
                         Operation *rhs, commutation_status status,
                         commutation_reason reason) {
    auto forward = analysis.getResult(lhs, rhs);
    auto reverse = analysis.getResult(rhs, lhs);
    EXPECT_EQ(forward.status, status);
    EXPECT_EQ(forward.reason, reason);
    EXPECT_EQ(reverse.status, status);
    EXPECT_EQ(reverse.reason, reason);
  }

  MLIRContext context;
};
} // namespace

TEST_F(CommutationAnalysisTest, CommutesOperationsOnDifferentQubits) {
  auto module = parseModule(R"mlir(
    module {
      func.func @disjoint() {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %x = quake.x %q0 : (!quake.wire) -> !quake.wire
        %h = quake.h %q1 : (!quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        quake.sink %h : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "disjoint");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 2u);
  CommutationAnalysis analysis(function.front());
  // X and H act on different virtual qubits.
  expectPair(analysis, operators[0], operators[1], commutation_status::commutes,
             commutation_reason::disjoint_support);
}

TEST_F(CommutationAnalysisTest, CanCommuteIsTrueOnlyForProvenCommutation) {
  auto module = parseModule(R"mlir(
    module {
      func.func @boolean_contract() {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %x = quake.x %q0 : (!quake.wire) -> !quake.wire
        %z = quake.z %x : (!quake.wire) -> !quake.wire
        %disjoint = quake.h %q1 : (!quake.wire) -> !quake.wire
        %h = quake.h %q2 : (!quake.wire) -> !quake.wire
        %s = quake.s %h : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        quake.sink %disjoint : !quake.wire
        quake.sink %s : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "boolean_contract");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 5u);
  CommutationAnalysis analysis(function.front());

  struct Query {
    Operation *lhs;
    Operation *rhs;
    commutation_status status;
    bool canCommute;
  };
  const Query queries[] = {
      {operators[0], operators[2], commutation_status::commutes, true},
      {operators[0], operators[1], commutation_status::does_not_commute, false},
      {operators[3], operators[4], commutation_status::indeterminate, false},
  };
  for (const auto &query : queries) {
    EXPECT_EQ(analysis.getResult(query.lhs, query.rhs).status, query.status);
    EXPECT_EQ(analysis.canCommute(query.lhs, query.rhs), query.canCommute);
  }
}

TEST_F(CommutationAnalysisTest, FreshReferenceProvenanceBoundaries) {
  auto module = parseModule(R"mlir(
    module {
      func.func @rebind(%destination: !quake.ref, %source: !quake.ref) {
        %wire = quake.unwrap %source : (!quake.ref) -> !quake.wire
        quake.wrap %wire to %destination : !quake.wire, !quake.ref
        return
      }

      func.func @fresh_allocations() {
        %angle = arith.constant 5.0e-1 : f64
        %first = quake.alloca !quake.ref
        %firstWire = quake.unwrap %first : (!quake.ref) -> !quake.wire
        %x = quake.x %firstWire : (!quake.wire) -> !quake.wire
        %rx = quake.rx (%angle) %x
            : (f64, !quake.wire) -> !quake.wire
        %second = quake.alloca !quake.ref
        %secondWire = quake.unwrap %second : (!quake.ref) -> !quake.wire
        %h = quake.h %secondWire : (!quake.wire) -> !quake.wire
        quake.sink %rx : !quake.wire
        quake.sink %h : !quake.wire
        return
      }

      func.func @call_boundary() {
        %destination = quake.alloca !quake.ref
        %destinationWire = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %destinationBefore = quake.x %destinationWire
            : (!quake.wire) -> !quake.wire
        quake.wrap %destinationBefore to %destination
            : !quake.wire, !quake.ref
        %source = quake.alloca !quake.ref
        %sourceWire = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        %sourceBefore = quake.h %sourceWire
            : (!quake.wire) -> !quake.wire
        quake.wrap %sourceBefore to %source : !quake.wire, !quake.ref
        func.call @rebind(%destination, %source)
            : (!quake.ref, !quake.ref) -> ()
        %destinationAfterWire = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %destinationAfter = quake.x %destinationAfterWire
            : (!quake.wire) -> !quake.wire
        %sourceAfterWire = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        %sourceAfter = quake.h %sourceAfterWire
            : (!quake.wire) -> !quake.wire
        quake.sink %destinationAfter : !quake.wire
        quake.sink %sourceAfter : !quake.wire
        return
      }

      func.func @region_boundary() {
        %destination = quake.alloca !quake.ref
        %destinationWire = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %destinationBefore = quake.x %destinationWire
            : (!quake.wire) -> !quake.wire
        quake.wrap %destinationBefore to %destination
            : !quake.wire, !quake.ref
        %source = quake.alloca !quake.ref
        %sourceWire = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        %sourceBefore = quake.h %sourceWire
            : (!quake.wire) -> !quake.wire
        quake.wrap %sourceBefore to %source : !quake.wire, !quake.ref
        cc.scope {
          %wire = quake.unwrap %source : (!quake.ref) -> !quake.wire
          quake.wrap %wire to %destination : !quake.wire, !quake.ref
          cc.continue
        }
        %destinationAfterWire = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %destinationAfter = quake.x %destinationAfterWire
            : (!quake.wire) -> !quake.wire
        %sourceAfterWire = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        %sourceAfter = quake.h %sourceAfterWire
            : (!quake.wire) -> !quake.wire
        quake.sink %destinationAfter : !quake.wire
        quake.sink %sourceAfter : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  auto fresh = getFunction(*module, "fresh_allocations");
  auto freshOperators = getOperators(fresh);
  ASSERT_EQ(freshOperators.size(), 3u);
  CommutationAnalysis freshAnalysis(fresh.front());
  expectPair(freshAnalysis, freshOperators[0], freshOperators[1],
             commutation_status::commutes, commutation_reason::same_axis);
  expectPair(freshAnalysis, freshOperators[0], freshOperators[2],
             commutation_status::commutes,
             commutation_reason::disjoint_support);

  for (llvm::StringRef functionName : {"call_boundary", "region_boundary"}) {
    auto boundary = getFunction(*module, functionName);
    auto boundaryOperators = getOperators(boundary);
    ASSERT_EQ(boundaryOperators.size(), 4u);
    CommutationAnalysis boundaryAnalysis(boundary.front());
    expectPair(boundaryAnalysis, boundaryOperators[0], boundaryOperators[1],
               commutation_status::commutes,
               commutation_reason::disjoint_support);
    expectPair(boundaryAnalysis, boundaryOperators[2], boundaryOperators[3],
               commutation_status::indeterminate,
               commutation_reason::unmapped_qubit_id);
  }
}

TEST_F(CommutationAnalysisTest, UnknownSuccessorArgumentSupport) {
  auto module = parseModule(R"mlir(
    module {
      func.func @successor_arguments() {
        %wire = quake.null_wire
        cf.br ^bb1(%wire, %wire : !quake.wire, !quake.wire)
      ^bb1(%first: !quake.wire, %second: !quake.wire):
        %x = quake.x %first : (!quake.wire) -> !quake.wire
        %h = quake.h %second : (!quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        quake.sink %h : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "successor_arguments");
  ASSERT_EQ(function.getBlocks().size(), 2u);
  Block &successor = function.back();
  auto operators =
      llvm::to_vector(successor.getOps<cudaq::quake::OperatorInterface>());
  ASSERT_EQ(operators.size(), 2u);

  CommutationAnalysis analysis(successor);
  expectPair(analysis, operators[0], operators[1],
             commutation_status::indeterminate,
             commutation_reason::unmapped_qubit_id);
}

TEST_F(CommutationAnalysisTest, CommutesMatchingOperationsAndAdjoints) {
  auto module = parseModule(R"mlir(
    module {
      func.func @same_operation(%theta: f64) {
        %zero0 = arith.constant 0.0 : f64
        %zero1 = arith.constant 0.0 : f64
        %one0 = arith.constant 1.0 : f64
        %one1 = arith.constant 1.0 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %q3 = quake.null_wire
        %rx0 = quake.rx (%theta) %q0 : (f64, !quake.wire) -> !quake.wire
        %rx1 = quake.rx<adj> (%theta) %rx0 : (f64, !quake.wire) -> !quake.wire
        %swap0:2 = quake.swap %rx1, %q1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %swap1:2 = quake.swap %swap0#1, %swap0#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %u20 = quake.u2 (%zero0, %one0) %q2 : (f64, f64, !quake.wire) -> !quake.wire
        %u21 = quake.u2 (%zero1, %one1) %u20 : (f64, f64, !quake.wire) -> !quake.wire
        %u30 = quake.u3 (%theta, %zero0, %one0) %q3 : (f64, f64, f64, !quake.wire) -> !quake.wire
        %u31 = quake.u3 (%theta, %zero0, %one0) %u30 : (f64, f64, f64, !quake.wire) -> !quake.wire
        quake.sink %swap1#0 : !quake.wire
        quake.sink %swap1#1 : !quake.wire
        quake.sink %u21 : !quake.wire
        quake.sink %u31 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "same_operation");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 8u);
  CommutationAnalysis analysis(function.front());
  // Rx and its adjoint have the same parameter and target.
  expectPair(analysis, operators[0], operators[1], commutation_status::commutes,
             commutation_reason::same_operation);
  // Reversing Swap's target order represents the same operation.
  expectPair(analysis, operators[2], operators[3], commutation_status::commutes,
             commutation_reason::same_operation);
  // Equal constant attributes make the two U2 parameter lists exact matches.
  expectPair(analysis, operators[4], operators[5], commutation_status::commutes,
             commutation_reason::same_operation);
  // The two U3 operations reuse the same SSA parameters and target.
  expectPair(analysis, operators[6], operators[7], commutation_status::commutes,
             commutation_reason::same_operation);
}

TEST_F(CommutationAnalysisTest, CommutesComputationalBasisDiagonalOperations) {
  auto module = parseModule(R"mlir(
    module {
      func.func @diagonal() {
        %angle = arith.constant 5.0e-1 : f64
        %q = quake.null_wire
        %z = quake.z %q : (!quake.wire) -> !quake.wire
        %s = quake.s %z : (!quake.wire) -> !quake.wire
        %t = quake.t %s : (!quake.wire) -> !quake.wire
        %r1 = quake.r1 (%angle) %t : (f64, !quake.wire) -> !quake.wire
        %rz = quake.rz (%angle) %r1 : (f64, !quake.wire) -> !quake.wire
        quake.sink %rz : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "diagonal");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 5u);
  CommutationAnalysis analysis(function.front());
  // Z and S are diagonal in the computational basis.
  expectPair(analysis, operators[0], operators[1], commutation_status::commutes,
             commutation_reason::computational_diagonal);
  // S and T are diagonal in the computational basis.
  expectPair(analysis, operators[1], operators[2], commutation_status::commutes,
             commutation_reason::computational_diagonal);
  // R1 and Rz are diagonal for any rotation angle.
  expectPair(analysis, operators[3], operators[4], commutation_status::commutes,
             commutation_reason::computational_diagonal);
}

TEST_F(CommutationAnalysisTest,
       CommutesMatchingAxesAndRejectsDifferentPhasedAxes) {
  auto module = parseModule(R"mlir(
    module {
      func.func @same_axis() {
        %angle0 = arith.constant 5.0e-1 : f64
        %angle1 = arith.constant 1.0 : f64
        %phase = arith.constant 2.5e-1 : f64
        %other_phase = arith.constant 7.5e-1 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %x = quake.x %q0 : (!quake.wire) -> !quake.wire
        %rx = quake.rx (%angle0) %x : (f64, !quake.wire) -> !quake.wire
        %p0 = quake.phased_rx (%angle0, %phase) %q1 : (f64, f64, !quake.wire) -> !quake.wire
        %p1 = quake.phased_rx (%angle1, %phase) %p0 : (f64, f64, !quake.wire) -> !quake.wire
        %p2 = quake.phased_rx (%angle0, %other_phase) %p1 : (f64, f64, !quake.wire) -> !quake.wire
        %y = quake.y %q2 : (!quake.wire) -> !quake.wire
        %ry = quake.ry (%angle0) %y : (f64, !quake.wire) -> !quake.wire
        quake.sink %rx : !quake.wire
        quake.sink %p2 : !quake.wire
        quake.sink %ry : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "same_axis");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 7u);
  CommutationAnalysis analysis(function.front());
  // X and Rx share the X axis on the same target.
  expectPair(analysis, operators[0], operators[1], commutation_status::commutes,
             commutation_reason::same_axis);
  // PhasedRx rotations share an axis when their phase parameters match.
  expectPair(analysis, operators[2], operators[3], commutation_status::commutes,
             commutation_reason::same_axis);
  // Different PhasedRx phase parameters do not establish a shared axis.
  expectPair(analysis, operators[3], operators[4],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
  // Y and Ry share the Y axis on the same target.
  expectPair(analysis, operators[5], operators[6], commutation_status::commutes,
             commutation_reason::same_axis);
}

TEST_F(CommutationAnalysisTest,
       ClassifiesPauliProductsByAnticommutationParity) {
  auto module = parseModule(R"mlir(
    module {
      func.func @pauli_parity(%word: !cc.charspan) {
        %angle = arith.constant 5.0e-1 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %q3 = quake.null_wire
        %q4 = quake.null_wire
        %xx:2 = quake.exp_pauli (%angle) %q0, %q1 to "XX" : (f64, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %zz:2 = quake.exp_pauli (%angle) %xx#0, %xx#1 to "ZZ" : (f64, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %x = quake.x %q2 : (!quake.wire) -> !quake.wire
        %z = quake.z %x : (!quake.wire) -> !quake.wire
        %exp_x = quake.exp_pauli (%angle) %q3 to "X" : (f64, !quake.wire) -> !quake.wire
        %exp_z = quake.z %exp_x : (!quake.wire) -> !quake.wire
        %dynamic = quake.exp_pauli (%angle) %q4 to %word : (f64, !quake.wire, !cc.charspan) -> !quake.wire
        %dynamic_z = quake.z %dynamic : (!quake.wire) -> !quake.wire
        quake.sink %zz#0 : !quake.wire
        quake.sink %zz#1 : !quake.wire
        quake.sink %z : !quake.wire
        quake.sink %exp_z : !quake.wire
        quake.sink %dynamic_z : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "pauli_parity");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 8u);
  CommutationAnalysis analysis(function.front());
  // XX and ZZ have two anti-commuting factors, giving even parity.
  expectPair(analysis, operators[0], operators[1], commutation_status::commutes,
             commutation_reason::even_pauli_parity);
  // X and Z have one anti-commuting factor and therefore do not commute.
  expectPair(analysis, operators[2], operators[3],
             commutation_status::does_not_commute,
             commutation_reason::odd_pauli_parity);
  // Odd parity does not prove that a parameterized ExpPauli rotation fails to
  // commute with Z for every angle.
  expectPair(analysis, operators[4], operators[5],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
  // A dynamic Pauli word cannot be normalized for comparison with Z.
  expectPair(analysis, operators[6], operators[7],
             commutation_status::indeterminate,
             commutation_reason::unsupported_pauli_word);
}

TEST_F(CommutationAnalysisTest, DiagonalOnControls) {
  auto module = parseModule(R"mlir(
    module {
      func.func @diagonal_on_controls() {
        %control = quake.null_wire
        %target = quake.null_wire
        %z = quake.z %control : (!quake.wire) -> !quake.wire
        %cx:2 = quake.x [%z] %target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %cx#0 : !quake.wire
        quake.sink %cx#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "diagonal_on_controls");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 2u);
  CommutationAnalysis analysis(function.front());
  // Z overlaps the controlled X only on its control qubit.
  expectPair(analysis, operators[0], operators[1], commutation_status::commutes,
             commutation_reason::diagonal_on_controls);
}

TEST_F(CommutationAnalysisTest, CompatibleControlledTargets) {
  auto module = parseModule(R"mlir(
    module {
      func.func @compatible_targets() {
        %angle = arith.constant 5.0e-1 : f64
        %control = quake.null_wire
        %target = quake.null_wire
        %cx:2 = quake.x [%control] %target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %crx:2 = quake.rx (%angle) [%cx#0] %cx#1 : (f64, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %cross_control = quake.null_wire
        %cross_target = quake.null_wire
        %cross_x:2 = quake.x [%cross_control] %cross_target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %cross_z:2 = quake.z [%cross_x#1] %cross_x#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %crx#0 : !quake.wire
        quake.sink %crx#1 : !quake.wire
        quake.sink %cross_z#0 : !quake.wire
        quake.sink %cross_z#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "compatible_targets");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 4u);
  CommutationAnalysis analysis(function.front());
  // Controlled X and Rx share a control and have compatible X-axis targets.
  expectPair(analysis, operators[0], operators[1], commutation_status::commutes,
             commutation_reason::compatible_controlled_targets);
  // Exchanging target and control roles prevents a controlled-target proof.
  expectPair(analysis, operators[2], operators[3],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
}

TEST_F(CommutationAnalysisTest, MutuallyExclusiveControls) {
  auto module = parseModule(R"mlir(
    module {
      func.func @exclusive_controls() {
        %control = quake.null_wire
        %target = quake.null_wire
        %x:2 = quake.x [%control] %target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %y:2 = quake.y [%x#0 neg [true]] %x#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %y#0 : !quake.wire
        quake.sink %y#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "exclusive_controls");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 2u);
  CommutationAnalysis analysis(function.front());
  // Opposite polarity on the shared control makes the predicates exclusive.
  expectPair(analysis, operators[0], operators[1], commutation_status::commutes,
             commutation_reason::mutually_exclusive_controls);
}

TEST_F(CommutationAnalysisTest, MeasurementInstrumentRelations) {
  auto module = parseModule(R"mlir(
    module {
      func.func @measurement() {
        %angle = arith.constant 5.0e-1 : f64
        %qx = quake.null_wire
        %x = quake.x %qx : (!quake.wire) -> !quake.wire
        %measurement, %measured = quake.mx %x
            : (!quake.wire) -> (!quake.measure, !quake.wire)
        %rx = quake.rx (%angle) %measured
            : (f64, !quake.wire) -> !quake.wire
        quake.sink %rx : !quake.wire
        %qy = quake.null_wire
        %y = quake.y %qy : (!quake.wire) -> !quake.wire
        %myMeasurement, %myMeasured = quake.my %y
            : (!quake.wire) -> (!quake.measure, !quake.wire)
        quake.sink %myMeasured : !quake.wire
        %qz = quake.null_wire
        %z = quake.z %qz : (!quake.wire) -> !quake.wire
        %mzMeasurement, %mzMeasured = quake.mz %z
            : (!quake.wire) -> (!quake.measure, !quake.wire)
        quake.sink %mzMeasured : !quake.wire
        %qxMz = quake.null_wire
        %xMz = quake.x %qxMz : (!quake.wire) -> !quake.wire
        %mzAfterX, %mzMeasuredAfterX = quake.mz %xMz
            : (!quake.wire) -> (!quake.measure, !quake.wire)
        quake.sink %mzMeasuredAfterX : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  auto measurement = getFunction(*module, "measurement");
  auto xOps = llvm::to_vector(measurement.front().getOps<cudaq::quake::XOp>());
  auto mxOps =
      llvm::to_vector(measurement.front().getOps<cudaq::quake::MxOp>());
  auto yOps = llvm::to_vector(measurement.front().getOps<cudaq::quake::YOp>());
  auto myOps =
      llvm::to_vector(measurement.front().getOps<cudaq::quake::MyOp>());
  auto zOps = llvm::to_vector(measurement.front().getOps<cudaq::quake::ZOp>());
  auto mzOps =
      llvm::to_vector(measurement.front().getOps<cudaq::quake::MzOp>());
  auto rxOps =
      llvm::to_vector(measurement.front().getOps<cudaq::quake::RxOp>());
  auto sinks =
      llvm::to_vector(measurement.front().getOps<cudaq::quake::SinkOp>());
  ASSERT_EQ(xOps.size(), 2u);
  ASSERT_EQ(mxOps.size(), 1u);
  ASSERT_EQ(yOps.size(), 1u);
  ASSERT_EQ(myOps.size(), 1u);
  ASSERT_EQ(zOps.size(), 1u);
  ASSERT_EQ(mzOps.size(), 2u);
  ASSERT_EQ(rxOps.size(), 1u);
  ASSERT_EQ(sinks.size(), 4u);

  CommutationAnalysis measurementAnalysis(measurement.front());
  // Mx and My return computational-basis wires, so moving the named-axis
  // operator after measurement would change the conditional output state.
  expectPair(measurementAnalysis, xOps[0], mxOps[0],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
  expectPair(measurementAnalysis, yOps[0], myOps[0],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
  // Mz already returns its wire in the basis measured by a Z-axis operator.
  expectPair(measurementAnalysis, zOps[0], mzOps[0],
             commutation_status::commutes,
             commutation_reason::measurement_instrument_basis);
  // X changes Mz's labelled conditional output state.
  expectPair(measurementAnalysis, xOps[1], mzOps[1],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
  // The later Rx retains the measured wire's block-local identity.
  expectPair(measurementAnalysis, xOps[0], rxOps[0],
             commutation_status::commutes, commutation_reason::same_axis);
  // A shared sink remains a conservative boundary.
  expectPair(measurementAnalysis, rxOps[0], sinks[0],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
}

TEST_F(CommutationAnalysisTest, ResetChannelRelations) {
  auto module = parseModule(R"mlir(
    module {
      func.func @reset() {
        %q = quake.null_wire
        %z = quake.z %q : (!quake.wire) -> !quake.wire
        %reset = quake.reset %z : (!quake.wire) -> !quake.wire
        %s = quake.s %reset : (!quake.wire) -> !quake.wire
        quake.sink %s : !quake.wire
        %qx = quake.null_wire
        %x = quake.x %qx : (!quake.wire) -> !quake.wire
        %resetAfterX = quake.reset %x : (!quake.wire) -> !quake.wire
        quake.sink %resetAfterX : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto reset = getFunction(*module, "reset");
  auto resetZOps = llvm::to_vector(reset.front().getOps<cudaq::quake::ZOp>());
  auto resetXOps = llvm::to_vector(reset.front().getOps<cudaq::quake::XOp>());
  auto resetOps =
      llvm::to_vector(reset.front().getOps<cudaq::quake::ResetOp>());
  auto sOps = llvm::to_vector(reset.front().getOps<cudaq::quake::SOp>());
  ASSERT_EQ(resetZOps.size(), 1u);
  ASSERT_EQ(resetXOps.size(), 1u);
  ASSERT_EQ(resetOps.size(), 2u);
  ASSERT_EQ(sOps.size(), 1u);

  CommutationAnalysis resetAnalysis(reset.front());
  // Reset commutes with operations that preserve the |0><0| state.
  expectPair(resetAnalysis, resetZOps[0], resetOps[0],
             commutation_status::commutes,
             commutation_reason::preserved_reset_state);
  // X does not preserve reset's |0><0| output state.
  expectPair(resetAnalysis, resetXOps[0], resetOps[1],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
  // The later S retains the reset wire's block-local identity.
  expectPair(resetAnalysis, resetZOps[0], sOps[0], commutation_status::commutes,
             commutation_reason::computational_diagonal);
}

TEST_F(CommutationAnalysisTest, UnsupportedMeasurementInstrumentBoundary) {
  auto module = parseModule(R"mlir(
    module {
      func.func @multi_target_measurement() {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %measurement, %measured:2 = quake.mz %q0, %q1
            : (!quake.wire, !quake.wire)
              -> (!cc.sequence<!cc.measure_handle>, !quake.wire, !quake.wire)
        %x = quake.x %q2 : (!quake.wire) -> !quake.wire
        quake.sink %measured#0 : !quake.wire
        quake.sink %measured#1 : !quake.wire
        quake.sink %x : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto multiTarget = getFunction(*module, "multi_target_measurement");
  auto multiMeasurements =
      llvm::to_vector(multiTarget.front().getOps<cudaq::quake::MzOp>());
  auto disjointXOps =
      llvm::to_vector(multiTarget.front().getOps<cudaq::quake::XOp>());
  ASSERT_EQ(multiMeasurements.size(), 1u);
  ASSERT_EQ(disjointXOps.size(), 1u);

  CommutationAnalysis multiTargetAnalysis(multiTarget.front());
  // Multi-target measurement instruments remain unsupported even on disjoint
  // support.
  expectPair(multiTargetAnalysis, multiMeasurements[0], disjointXOps[0],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
}

TEST_F(CommutationAnalysisTest,
       CommutesMatchingCustomUnitariesAndRejectsOpaqueDifferences) {
  auto module = parseModule(R"mlir(
    module {
      func.func private @unitary_generator()
      func.func private @other_unitary_generator()
      func.func @opaque_unitaries() {
        %angle0 = arith.constant 2.5e-1 : f64
        %angle1 = arith.constant 5.0e-1 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %u0 = quake.custom_unitary_call @unitary_generator %q0 : (!quake.wire) -> !quake.wire
        %u1 = quake.custom_unitary_call @unitary_generator %q1 : (!quake.wire) -> !quake.wire
        %u2 = quake.custom_unitary_call @unitary_generator<adj> %u0 : (!quake.wire) -> !quake.wire
        %u3 = quake.custom_unitary_call @other_unitary_generator %u2 : (!quake.wire) -> !quake.wire
        %u4 = quake.custom_unitary_call @unitary_generator(%angle0) %q2 : (f64, !quake.wire) -> !quake.wire
        %u5 = quake.custom_unitary_call @unitary_generator(%angle1) %u4 : (f64, !quake.wire) -> !quake.wire
        quake.sink %u1 : !quake.wire
        quake.sink %u3 : !quake.wire
        quake.sink %u5 : !quake.wire
        return
      }
      func.func @constant_unitaries() {
        %q = quake.null_wire
        %u0 = quake.custom_unitary_constant @unitary_matrix %q : (!quake.wire) -> !quake.wire
        %u1 = quake.custom_unitary_constant @unitary_matrix<adj> %u0 : (!quake.wire) -> !quake.wire
        quake.sink %u1 : !quake.wire
        return
      }
      cc.global constant private @unitary_matrix (dense<[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
    })mlir");
  ASSERT_TRUE(module);
  auto opaque = getFunction(*module, "opaque_unitaries");
  auto opaqueOperators = getOperators(opaque);
  ASSERT_EQ(opaqueOperators.size(), 6u);
  CommutationAnalysis opaqueAnalysis(opaque.front());
  // Opaque custom unitaries still commute when their supports are disjoint.
  expectPair(opaqueAnalysis, opaqueOperators[0], opaqueOperators[1],
             commutation_status::commutes,
             commutation_reason::disjoint_support);
  // A custom unitary commutes with its adjoint when the definition matches.
  expectPair(opaqueAnalysis, opaqueOperators[0], opaqueOperators[2],
             commutation_status::commutes, commutation_reason::same_operation);
  // Different custom-unitary definitions remain opaque on shared support.
  expectPair(opaqueAnalysis, opaqueOperators[0], opaqueOperators[3],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);
  // Unequal parameters prevent a same-operation proof for one definition.
  expectPair(opaqueAnalysis, opaqueOperators[4], opaqueOperators[5],
             commutation_status::indeterminate,
             commutation_reason::no_applicable_rule);

  auto constant = getFunction(*module, "constant_unitaries");
  auto constantOperators = getOperators(constant);
  ASSERT_EQ(constantOperators.size(), 2u);
  CommutationAnalysis constantAnalysis(constant.front());
  // Constant custom unitaries share the same matrix symbol and target.
  expectPair(constantAnalysis, constantOperators[0], constantOperators[1],
             commutation_status::commutes, commutation_reason::same_operation);
}

TEST_F(CommutationAnalysisTest,
       ReturnsIndeterminateForUnsupportedOrUnresolvedQueries) {
  auto module = parseModule(R"mlir(
    module {
      func.func private @wire_source() -> !quake.wire
      func.func @unsupported_query() {
        %q = quake.null_wire
        %x = quake.x %q : (!quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        return
      }
      func.func @aggregate(%q: !quake.veq<2>) {
        quake.x %q : (!quake.veq<2>) -> ()
        return
      }
      func.func @reusable_control() {
        %control_wire = quake.null_wire
        %target0 = quake.null_wire
        %target1 = quake.null_wire
        %control = quake.to_ctrl %control_wire : (!quake.wire) -> !quake.control
        %x0 = quake.x [%control] %target0 : (!quake.control, !quake.wire) -> !quake.wire
        %x1 = quake.x [%control] %target1 : (!quake.control, !quake.wire) -> !quake.wire
        %returned = quake.from_ctrl %control : (!quake.control) -> !quake.wire
        quake.sink %returned : !quake.wire
        quake.sink %x0 : !quake.wire
        quake.sink %x1 : !quake.wire
        return
      }
      func.func @call_result() {
        %q = call @wire_source() : () -> !quake.wire
        %x = quake.x %q : (!quake.wire) -> !quake.wire
        %z = quake.z %x : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @different_failures(%aggregate: !quake.veq<2>) {
        %q = call @wire_source() : () -> !quake.wire
        quake.x %aggregate : (!quake.veq<2>) -> ()
        %x = quake.x %q : (!quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        return
      }
      func.func @other() {
        %q = quake.null_wire
        %z = quake.z %q : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  auto function = getFunction(*module, "unsupported_query");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 1u);
  auto *returnOp = function.front().getTerminator();
  CommutationAnalysis analysis(function.front());
  // A null query operand cannot be analyzed.
  expectPair(analysis, nullptr, operators[0], commutation_status::indeterminate,
             commutation_reason::null_operation);
  // func.return does not implement Quake OperatorInterface.
  expectPair(analysis, operators[0], returnOp,
             commutation_status::indeterminate,
             commutation_reason::unsupported_operation_kind);

  auto aggregate = getFunction(*module, "aggregate");
  auto aggregateOperators = getOperators(aggregate);
  ASSERT_EQ(aggregateOperators.size(), 1u);
  CommutationAnalysis aggregateAnalysis(aggregate.front());
  // Aggregate targets are outside the supported scalar value form.
  expectPair(aggregateAnalysis, aggregateOperators[0], aggregateOperators[0],
             commutation_status::indeterminate,
             commutation_reason::unsupported_quantum_operand_type);

  auto reusableControl = getFunction(*module, "reusable_control");
  auto reusableControlOperators = getOperators(reusableControl);
  ASSERT_EQ(reusableControlOperators.size(), 2u);
  CommutationAnalysis reusableControlAnalysis(reusableControl.front());
  // Reusable controls are valid Quake but outside the scalar wire contract.
  expectPair(reusableControlAnalysis, reusableControlOperators[0],
             reusableControlOperators[1], commutation_status::indeterminate,
             commutation_reason::unsupported_quantum_operand_type);

  auto callResult = getFunction(*module, "call_result");
  auto callOperators = getOperators(callResult);
  ASSERT_EQ(callOperators.size(), 2u);
  CommutationAnalysis callAnalysis(callResult.front());
  // Qubit identity is not propagated through a function call result.
  expectPair(callAnalysis, callOperators[0], callOperators[1],
             commutation_status::indeterminate,
             commutation_reason::unmapped_qubit_id);

  auto differentFailures = getFunction(*module, "different_failures");
  auto failureOperators = getOperators(differentFailures);
  ASSERT_EQ(failureOperators.size(), 2u);
  CommutationAnalysis forwardAnalysis(differentFailures.front());
  CommutationAnalysis reverseAnalysis(differentFailures.front());
  // Canonical evaluation produces one detailed result independent of which
  // query order first populates an analysis instance.
  auto forward =
      forwardAnalysis.getResult(failureOperators[0], failureOperators[1]);
  auto reverse =
      reverseAnalysis.getResult(failureOperators[1], failureOperators[0]);
  EXPECT_EQ(forward.status, reverse.status);
  EXPECT_EQ(forward.reason, reverse.reason);

  auto other = getFunction(*module, "other");
  auto otherOperators = getOperators(other);
  ASSERT_EQ(otherOperators.size(), 1u);
  // Operations outside the block owned by the analysis cannot be compared.
  expectPair(analysis, operators[0], otherOperators[0],
             commutation_status::indeterminate,
             commutation_reason::different_blocks);
}

TEST(CommutationReasonTest, ReturnsStableIdentifierForEveryReason) {
  struct ReasonCase {
    commutation_reason reason;
    llvm::StringLiteral identifier;
  };
  static constexpr ReasonCase cases[] = {
      {commutation_reason::disjoint_support, "disjoint-support"},
      {commutation_reason::same_operation, "same-operation"},
      {commutation_reason::computational_diagonal, "computational-diagonal"},
      {commutation_reason::same_axis, "same-axis"},
      {commutation_reason::measurement_instrument_basis,
       "measurement-instrument-basis"},
      {commutation_reason::preserved_reset_state, "preserved-reset-state"},
      {commutation_reason::even_pauli_parity, "even-pauli-parity"},
      {commutation_reason::diagonal_on_controls, "diagonal-on-controls"},
      {commutation_reason::compatible_controlled_targets,
       "compatible-controlled-targets"},
      {commutation_reason::mutually_exclusive_controls,
       "mutually-exclusive-controls"},
      {commutation_reason::odd_pauli_parity, "odd-pauli-parity"},
      {commutation_reason::null_operation, "null-operation"},
      {commutation_reason::different_blocks, "different-blocks"},
      {commutation_reason::unsupported_operation_kind,
       "unsupported-operation-kind"},
      {commutation_reason::unsupported_quantum_operand_type,
       "unsupported-quantum-operand-type"},
      {commutation_reason::unmapped_qubit_id, "unmapped-qubit-id"},
      {commutation_reason::duplicate_qubit_operand, "duplicate-qubit-operand"},
      {commutation_reason::unsupported_pauli_word, "unsupported-pauli-word"},
      {commutation_reason::no_applicable_rule, "no-applicable-rule"},
  };

  for (const auto &testCase : cases)
    EXPECT_EQ(cudaq::quake::detail::getCommutationReasonId(testCase.reason),
              testCase.identifier);
}
