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
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;

using cudaq::quake::detail::CommutationAnalysis;
using cudaq::quake::detail::CommutationReason;
using cudaq::quake::detail::CommutationStatus;

namespace {
class CommutationAnalysisTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect>();
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

  // Check that both operand orders produce the expected detailed result and
  // that the boolean convenience query agrees with the commutation status.
  static void expectPair(CommutationAnalysis &analysis, Operation *lhs,
                         Operation *rhs, CommutationStatus status,
                         CommutationReason reason) {
    auto forward = analysis.getResult(lhs, rhs);
    auto reverse = analysis.getResult(rhs, lhs);
    EXPECT_EQ(forward.status, status);
    EXPECT_EQ(forward.reason, reason);
    EXPECT_EQ(reverse.status, status);
    EXPECT_EQ(reverse.reason, reason);
    EXPECT_EQ(analysis.canCommute(lhs, rhs),
              status == CommutationStatus::Commutes);
    EXPECT_EQ(analysis.canCommute(rhs, lhs),
              status == CommutationStatus::Commutes);
  }

  MLIRContext context;
};
} // namespace

TEST_F(CommutationAnalysisTest, DisjointSupport) {
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
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::DisjointSupport);
}

TEST_F(CommutationAnalysisTest, SameOperation) {
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
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  // Reversing Swap's target order represents the same operation.
  expectPair(analysis, operators[2], operators[3], CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  // Equal constant attributes make the two U2 parameter lists exact matches.
  expectPair(analysis, operators[4], operators[5], CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  // The two U3 operations reuse the same SSA parameters and target.
  expectPair(analysis, operators[6], operators[7], CommutationStatus::Commutes,
             CommutationReason::SameOperation);
}

TEST_F(CommutationAnalysisTest, ComputationalDiagonal) {
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
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::ComputationalDiagonal);
  // S and T are diagonal in the computational basis.
  expectPair(analysis, operators[1], operators[2], CommutationStatus::Commutes,
             CommutationReason::ComputationalDiagonal);
  // R1 and Rz are diagonal for any rotation angle.
  expectPair(analysis, operators[3], operators[4], CommutationStatus::Commutes,
             CommutationReason::ComputationalDiagonal);
}

TEST_F(CommutationAnalysisTest, SameAxis) {
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
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::SameAxis);
  // PhasedRx rotations share an axis when their phase parameters match.
  expectPair(analysis, operators[2], operators[3], CommutationStatus::Commutes,
             CommutationReason::SameAxis);
  // Different PhasedRx phase parameters do not establish a shared axis.
  expectPair(analysis, operators[3], operators[4],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
  // Y and Ry share the Y axis on the same target.
  expectPair(analysis, operators[5], operators[6], CommutationStatus::Commutes,
             CommutationReason::SameAxis);
}

TEST_F(CommutationAnalysisTest, PauliParity) {
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
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::EvenPauliParity);
  // X and Z have one anti-commuting factor and therefore do not commute.
  expectPair(analysis, operators[2], operators[3],
             CommutationStatus::DoesNotCommute,
             CommutationReason::OddPauliParity);
  // Odd parity does not prove that a parameterized ExpPauli rotation fails to
  // commute with Z for every angle.
  expectPair(analysis, operators[4], operators[5],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
  // A dynamic Pauli word cannot be normalized for comparison with Z.
  expectPair(analysis, operators[6], operators[7],
             CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedPauliWord);
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
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::DiagonalOnControls);
}

TEST_F(CommutationAnalysisTest, CompatibleControlledTargets) {
  auto module = parseModule(R"mlir(
    module {
      func.func @compatible_targets() {
        %angle = arith.constant 5.0e-1 : f64
        %control_wire = quake.null_wire
        %target = quake.null_wire
        %control = quake.to_ctrl %control_wire : (!quake.wire) -> !quake.control
        %cx = quake.x [%control] %target : (!quake.control, !quake.wire) -> !quake.wire
        %crx = quake.rx (%angle) [%control] %cx : (f64, !quake.control, !quake.wire) -> !quake.wire
        %cross_control = quake.null_wire
        %cross_target = quake.null_wire
        %cross_x:2 = quake.x [%cross_control] %cross_target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %cross_z:2 = quake.z [%cross_x#1] %cross_x#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %crx : !quake.wire
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
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::CompatibleControlledTargets);
  // Exchanging target and control roles prevents a controlled-target proof.
  expectPair(analysis, operators[2], operators[3],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
}

TEST_F(CommutationAnalysisTest, MutuallyExclusiveControls) {
  auto module = parseModule(R"mlir(
    module {
      func.func @exclusive_controls() {
        %control_wire = quake.null_wire
        %target = quake.null_wire
        %control = quake.to_ctrl %control_wire : (!quake.wire) -> !quake.control
        %x = quake.x [%control] %target : (!quake.control, !quake.wire) -> !quake.wire
        %y = quake.y [%control neg [true]] %x : (!quake.control, !quake.wire) -> !quake.wire
        quake.sink %y : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "exclusive_controls");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 2u);
  CommutationAnalysis analysis(function.front());
  // Opposite polarity on the shared control makes the predicates exclusive.
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::MutuallyExclusiveControls);
}

TEST_F(CommutationAnalysisTest, CustomUnitaryRules) {
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
             CommutationStatus::Commutes, CommutationReason::DisjointSupport);
  // A custom unitary commutes with its adjoint when the definition matches.
  expectPair(opaqueAnalysis, opaqueOperators[0], opaqueOperators[2],
             CommutationStatus::Commutes, CommutationReason::SameOperation);
  // Different custom-unitary definitions remain opaque on shared support.
  expectPair(opaqueAnalysis, opaqueOperators[0], opaqueOperators[3],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
  // Unequal parameters prevent a same-operation proof for one definition.
  expectPair(opaqueAnalysis, opaqueOperators[4], opaqueOperators[5],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);

  auto constant = getFunction(*module, "constant_unitaries");
  auto constantOperators = getOperators(constant);
  ASSERT_EQ(constantOperators.size(), 2u);
  CommutationAnalysis constantAnalysis(constant.front());
  // Constant custom unitaries share the same matrix symbol and target.
  expectPair(constantAnalysis, constantOperators[0], constantOperators[1],
             CommutationStatus::Commutes, CommutationReason::SameOperation);
}

TEST_F(CommutationAnalysisTest, UnsupportedQueries) {
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
      func.func @duplicate_role() {
        %control_wire = quake.null_wire
        %target = quake.null_wire
        %control = quake.to_ctrl %control_wire : (!quake.wire) -> !quake.control
        %x = quake.x [%control, %control] %target : (!quake.control, !quake.control, !quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
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
  expectPair(analysis, nullptr, operators[0], CommutationStatus::Indeterminate,
             CommutationReason::NullOperation);
  // func.return does not implement Quake OperatorInterface.
  expectPair(analysis, operators[0], returnOp, CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedOperationKind);

  auto aggregate = getFunction(*module, "aggregate");
  auto aggregateOperators = getOperators(aggregate);
  ASSERT_EQ(aggregateOperators.size(), 1u);
  CommutationAnalysis aggregateAnalysis(aggregate.front());
  // Aggregate targets are outside the supported scalar value form.
  expectPair(aggregateAnalysis, aggregateOperators[0], aggregateOperators[0],
             CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedQuantumOperandType);

  auto duplicate = getFunction(*module, "duplicate_role");
  auto duplicateOperators = getOperators(duplicate);
  ASSERT_EQ(duplicateOperators.size(), 1u);
  CommutationAnalysis duplicateAnalysis(duplicate.front());
  // One virtual qubit cannot occupy two positions in a supported operation.
  expectPair(duplicateAnalysis, duplicateOperators[0], duplicateOperators[0],
             CommutationStatus::Indeterminate,
             CommutationReason::DuplicateQubitOperand);

  auto callResult = getFunction(*module, "call_result");
  auto callOperators = getOperators(callResult);
  ASSERT_EQ(callOperators.size(), 2u);
  CommutationAnalysis callAnalysis(callResult.front());
  // Qubit identity is not propagated through a function call result.
  expectPair(callAnalysis, callOperators[0], callOperators[1],
             CommutationStatus::Indeterminate,
             CommutationReason::UnmappedQubitId);

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
             CommutationStatus::Indeterminate,
             CommutationReason::DifferentBlocks);
}
