/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "QubitIdentityAnalysis.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <string>

using namespace mlir;

using CommutationAnalysis = cudaq::quake::detail::CommutationAnalysis;
using commutation_reason = cudaq::quake::detail::commutation_reason;
using commutation_status = cudaq::quake::detail::commutation_status;
using QubitIdentityAnalysis = cudaq::quake::detail::QubitIdentityAnalysis;

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
    for (Operation &operation : function.front())
      if (isa<cudaq::quake::OperatorInterface>(operation))
        operators.push_back(&operation);
    return operators;
  }

  static void expectResult(CommutationAnalysis &analysis, Operation *lhs,
                           Operation *rhs, commutation_status status,
                           commutation_reason reason) {
    auto result = analysis.getResult(lhs, rhs);
    EXPECT_EQ(result.status, status);
    EXPECT_EQ(result.reason, reason);
  }

  MLIRContext context;
};
} // namespace

TEST_F(CommutationAnalysisTest, ReportsCoreDetailedResultsSymmetrically) {
  auto module = parseModule(R"mlir(
    module {
      func.func @relations() {
        %q0 = quake.null_wire
        %x0 = quake.x %q0 : (!quake.wire) -> !quake.wire
        %z0 = quake.z %x0 : (!quake.wire) -> !quake.wire
        %q1 = quake.null_wire
        %h1 = quake.h %q1 : (!quake.wire) -> !quake.wire
        %q2 = quake.null_wire
        %h2 = quake.h %q2 : (!quake.wire) -> !quake.wire
        %s2 = quake.s %h2 : (!quake.wire) -> !quake.wire
        %q3 = quake.null_wire
        %x3 = quake.x %q3 : (!quake.wire) -> !quake.wire
        quake.sink %z0 : !quake.wire
        quake.sink %h1 : !quake.wire
        quake.sink %s2 : !quake.wire
        quake.sink %x3 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "relations");
  auto operators = getOperators(function);
  auto sinks = llvm::to_vector(function.getOps<cudaq::quake::SinkOp>());
  ASSERT_EQ(operators.size(), 6u);
  ASSERT_EQ(sinks.size(), 4u);
  CommutationAnalysis analysis(function.front());

  expectResult(analysis, operators[0], operators[2],
               commutation_status::commutes,
               commutation_reason::disjoint_support);
  expectResult(analysis, operators[2], operators[0],
               commutation_status::commutes,
               commutation_reason::disjoint_support);
  EXPECT_TRUE(analysis.canCommute(operators[0], operators[2]));

  expectResult(analysis, operators[0], operators[1],
               commutation_status::does_not_commute,
               commutation_reason::odd_pauli_parity);
  expectResult(analysis, operators[1], operators[0],
               commutation_status::does_not_commute,
               commutation_reason::odd_pauli_parity);
  EXPECT_FALSE(analysis.canCommute(operators[0], operators[1]));

  expectResult(analysis, operators[3], operators[4],
               commutation_status::indeterminate,
               commutation_reason::no_applicable_rule);
  EXPECT_FALSE(analysis.canCommute(operators[3], operators[4]));
  expectResult(analysis, operators[5], sinks[3],
               commutation_status::indeterminate,
               commutation_reason::no_applicable_rule);
}

TEST_F(CommutationAnalysisTest, ClassifiesRepresentativeSemanticRules) {
  auto module = parseModule(R"mlir(
    module {
      quake.wire_set @wires[1]
      func.func @diagonal() {
        %q = quake.null_wire
        %z = quake.z %q : (!quake.wire) -> !quake.wire
        %s = quake.s %z : (!quake.wire) -> !quake.wire
        quake.sink %s : !quake.wire
        return
      }
      func.func @axes() {
        %angle0 = arith.constant 5.0e-1 : f64
        %angle1 = arith.constant 1.0 : f64
        %phase0 = arith.constant 2.5e-1 : f64
        %phase1 = arith.constant 7.5e-1 : f64
        %q0 = quake.null_wire
        %x = quake.x %q0 : (!quake.wire) -> !quake.wire
        %rx = quake.rx (%angle0) %x : (f64, !quake.wire) -> !quake.wire
        %q1 = quake.null_wire
        %p0 = quake.phased_rx (%angle0, %phase0) %q1
            : (f64, f64, !quake.wire) -> !quake.wire
        %p1 = quake.phased_rx (%angle1, %phase0) %p0
            : (f64, f64, !quake.wire) -> !quake.wire
        %p2 = quake.phased_rx (%angle1, %phase1) %p1
            : (f64, f64, !quake.wire) -> !quake.wire
        quake.sink %rx : !quake.wire
        quake.sink %p2 : !quake.wire
        return
      }
      func.func @paulis(%word: !cc.charspan) {
        %angle = arith.constant 5.0e-1 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %xx:2 = quake.exp_pauli (%angle) %q0, %q1 to "XX"
            : (f64, !quake.wire, !quake.wire)
              -> (!quake.wire, !quake.wire)
        %zz:2 = quake.exp_pauli (%angle) %xx#0, %xx#1 to "ZZ"
            : (f64, !quake.wire, !quake.wire)
              -> (!quake.wire, !quake.wire)
        %q2 = quake.null_wire
        %dynamic = quake.exp_pauli (%angle) %q2 to %word
            : (f64, !quake.wire, !cc.charspan) -> !quake.wire
        %z = quake.z %dynamic : (!quake.wire) -> !quake.wire
        quake.sink %zz#0 : !quake.wire
        quake.sink %zz#1 : !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @diagonal_control() {
        %control = quake.null_wire
        %target = quake.null_wire
        %z = quake.z %control : (!quake.wire) -> !quake.wire
        %cx:2 = quake.x [%z] %target
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %cx#0 : !quake.wire
        quake.sink %cx#1 : !quake.wire
        return
      }
      func.func @controlled_targets() {
        %angle = arith.constant 5.0e-1 : f64
        %control = quake.null_wire
        %target = quake.null_wire
        %cx:2 = quake.x [%control] %target
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %crx:2 = quake.rx (%angle) [%cx#0] %cx#1
            : (f64, !quake.wire, !quake.wire)
              -> (!quake.wire, !quake.wire)
        %crossControl = quake.null_wire
        %crossTarget = quake.null_wire
        %crossX:2 = quake.x [%crossControl] %crossTarget
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %crossZ:2 = quake.z [%crossX#1] %crossX#0
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %crx#0 : !quake.wire
        quake.sink %crx#1 : !quake.wire
        quake.sink %crossZ#0 : !quake.wire
        quake.sink %crossZ#1 : !quake.wire
        return
      }
      func.func @exclusive_controls() {
        %control = quake.null_wire
        %target = quake.null_wire
        %x:2 = quake.x [%control] %target
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %y:2 = quake.y [%x#0 neg [true]] %x#1
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %y#0 : !quake.wire
        quake.sink %y#1 : !quake.wire
        return
      }
      func.func @measurement() {
        %qx = quake.null_wire
        %x = quake.x %qx : (!quake.wire) -> !quake.wire
        %mx, %mxWire = quake.mx %x
            : (!quake.wire) -> (!quake.measure, !quake.wire)
        %qz = quake.null_wire
        %z = quake.z %qz : (!quake.wire) -> !quake.wire
        %mz, %mzWire = quake.mz %z
            : (!quake.wire) -> (!quake.measure, !quake.wire)
        quake.sink %mxWire : !quake.wire
        quake.sink %mzWire : !quake.wire
        return
      }
      func.func @reset() {
        %qz = quake.null_wire
        %z = quake.z %qz : (!quake.wire) -> !quake.wire
        %resetZ = quake.reset %z : (!quake.wire) -> !quake.wire
        %qx = quake.null_wire
        %x = quake.x %qx : (!quake.wire) -> !quake.wire
        %resetX = quake.reset %x : (!quake.wire) -> !quake.wire
        quake.sink %resetZ : !quake.wire
        quake.sink %resetX : !quake.wire
        return
      }
      func.func @duplicate_operand() {
        %control = quake.borrow_wire @wires[0] : !quake.wire
        %target = quake.borrow_wire @wires[0] : !quake.wire
        %result:2 = quake.x [%control] %target
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.return_wire %result#0 : !quake.wire
        quake.return_wire %result#1 : !quake.wire
        return
      }
      func.func @reusable_control() {
        %wire = quake.null_wire
        %control = quake.to_ctrl %wire : (!quake.wire) -> !quake.control
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %x0 = quake.x [%control] %q0
            : (!quake.control, !quake.wire) -> !quake.wire
        %x1 = quake.x [%control] %q1
            : (!quake.control, !quake.wire) -> !quake.wire
        %returned = quake.from_ctrl %control
            : (!quake.control) -> !quake.wire
        quake.sink %returned : !quake.wire
        quake.sink %x0 : !quake.wire
        quake.sink %x1 : !quake.wire
        return
      }
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

  auto checkOperators = [&](llvm::StringRef functionName, unsigned lhs,
                            unsigned rhs, commutation_status status,
                            commutation_reason reason) {
    auto function = getFunction(*module, functionName);
    auto operators = getOperators(function);
    ASSERT_LT(lhs, operators.size());
    ASSERT_LT(rhs, operators.size());
    CommutationAnalysis analysis(function.front());
    expectResult(analysis, operators[lhs], operators[rhs], status, reason);
  };
  checkOperators("diagonal", 0, 1, commutation_status::commutes,
                 commutation_reason::computational_diagonal);
  checkOperators("axes", 0, 1, commutation_status::commutes,
                 commutation_reason::same_axis);
  checkOperators("axes", 2, 3, commutation_status::commutes,
                 commutation_reason::same_axis);
  checkOperators("axes", 3, 4, commutation_status::indeterminate,
                 commutation_reason::no_applicable_rule);
  checkOperators("paulis", 0, 1, commutation_status::commutes,
                 commutation_reason::even_pauli_parity);
  checkOperators("paulis", 2, 3, commutation_status::indeterminate,
                 commutation_reason::unsupported_pauli_word);
  checkOperators("diagonal_control", 0, 1, commutation_status::commutes,
                 commutation_reason::diagonal_on_controls);
  checkOperators("controlled_targets", 0, 1, commutation_status::commutes,
                 commutation_reason::compatible_controlled_targets);
  checkOperators("controlled_targets", 2, 3, commutation_status::indeterminate,
                 commutation_reason::no_applicable_rule);
  checkOperators("exclusive_controls", 0, 1, commutation_status::commutes,
                 commutation_reason::mutually_exclusive_controls);
  checkOperators("duplicate_operand", 0, 0, commutation_status::indeterminate,
                 commutation_reason::duplicate_qubit_operand);
  checkOperators("reusable_control", 0, 1, commutation_status::indeterminate,
                 commutation_reason::unsupported_quantum_operand_type);

  auto measurement = getFunction(*module, "measurement");
  auto measurementOperators = getOperators(measurement);
  auto mx = *measurement.getOps<cudaq::quake::MxOp>().begin();
  auto mz = *measurement.getOps<cudaq::quake::MzOp>().begin();
  ASSERT_EQ(measurementOperators.size(), 2u);
  CommutationAnalysis measurementAnalysis(measurement.front());
  expectResult(measurementAnalysis, measurementOperators[0], mx,
               commutation_status::indeterminate,
               commutation_reason::no_applicable_rule);
  expectResult(measurementAnalysis, measurementOperators[1], mz,
               commutation_status::commutes,
               commutation_reason::measurement_instrument_basis);

  auto reset = getFunction(*module, "reset");
  auto resetOperators = getOperators(reset);
  auto resets = llvm::to_vector(reset.getOps<cudaq::quake::ResetOp>());
  ASSERT_EQ(resetOperators.size(), 2u);
  ASSERT_EQ(resets.size(), 2u);
  CommutationAnalysis resetAnalysis(reset.front());
  expectResult(resetAnalysis, resetOperators[0], resets[0],
               commutation_status::commutes,
               commutation_reason::preserved_reset_state);
  expectResult(resetAnalysis, resetOperators[1], resets[1],
               commutation_status::indeterminate,
               commutation_reason::no_applicable_rule);

  auto multiTarget = getFunction(*module, "multi_target_measurement");
  auto multiTargetOperators = getOperators(multiTarget);
  auto multiTargetMz = *multiTarget.getOps<cudaq::quake::MzOp>().begin();
  ASSERT_EQ(multiTargetOperators.size(), 1u);
  CommutationAnalysis multiTargetAnalysis(multiTarget.front());
  expectResult(multiTargetAnalysis, multiTargetMz, multiTargetOperators[0],
               commutation_status::indeterminate,
               commutation_reason::no_applicable_rule);
}

TEST_F(CommutationAnalysisTest, NormalizesValidationFailuresBeforeCaching) {
  auto module = parseModule(R"mlir(
    module {
      quake.wire_set @wires[1]
      func.func private @wire_source() -> !quake.wire
      func.func @failures(%aggregate: !quake.veq<2>) {
        %unknown = func.call @wire_source() : () -> !quake.wire
        quake.x %aggregate : (!quake.veq<2>) -> ()
        %unmapped = quake.x %unknown : (!quake.wire) -> !quake.wire
        %control = quake.borrow_wire @wires[0] : !quake.wire
        %target = quake.borrow_wire @wires[0] : !quake.wire
        %duplicate:2 = quake.x [%control] %target
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %unmapped : !quake.wire
        quake.return_wire %duplicate#0 : !quake.wire
        quake.return_wire %duplicate#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "failures");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 3u);

  CommutationAnalysis unsupportedFirst(function.front());
  CommutationAnalysis unmappedFirst(function.front());
  expectResult(unsupportedFirst, operators[0], operators[1],
               commutation_status::indeterminate,
               commutation_reason::unsupported_quantum_operand_type);
  expectResult(unmappedFirst, operators[1], operators[0],
               commutation_status::indeterminate,
               commutation_reason::unsupported_quantum_operand_type);

  CommutationAnalysis unmappedBeforeDuplicate(function.front());
  CommutationAnalysis duplicateBeforeUnmapped(function.front());
  expectResult(unmappedBeforeDuplicate, operators[1], operators[2],
               commutation_status::indeterminate,
               commutation_reason::unmapped_qubit_id);
  expectResult(duplicateBeforeUnmapped, operators[2], operators[1],
               commutation_status::indeterminate,
               commutation_reason::unmapped_qubit_id);
}

TEST_F(CommutationAnalysisTest, RejectsInvalidQueries) {
  auto module = parseModule(R"mlir(
    module {
      func.func @local() {
        %q = quake.null_wire
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
  auto local = getFunction(*module, "local");
  auto other = getFunction(*module, "other");
  auto localOperators = getOperators(local);
  auto otherOperators = getOperators(other);
  ASSERT_EQ(localOperators.size(), 1u);
  ASSERT_EQ(otherOperators.size(), 1u);
  CommutationAnalysis analysis(local.front());

  expectResult(analysis, nullptr, localOperators[0],
               commutation_status::indeterminate,
               commutation_reason::null_operation);
  expectResult(analysis, localOperators[0], local.front().getTerminator(),
               commutation_status::indeterminate,
               commutation_reason::unsupported_operation_kind);
  expectResult(analysis, localOperators[0], otherOperators[0],
               commutation_status::indeterminate,
               commutation_reason::different_blocks);
}

TEST_F(CommutationAnalysisTest, RecognizesMatchingCustomUnitaryDefinitions) {
  auto module = parseModule(R"mlir(
    module {
      func.func private @generator()
      func.func private @other_generator()
      func.func @custom_unitaries() {
        %angle0 = arith.constant 2.5e-1 : f64
        %angle0Equal = arith.constant 2.5e-1 : f64
        %angle1 = arith.constant 5.0e-1 : f64
        %q0 = quake.null_wire
        %call0 = quake.custom_unitary_call @generator %q0
            : (!quake.wire) -> !quake.wire
        %call1 = quake.custom_unitary_call @generator<adj> %call0
            : (!quake.wire) -> !quake.wire
        %opaque = quake.custom_unitary_call @other_generator %call1
            : (!quake.wire) -> !quake.wire
        %q1 = quake.null_wire
        %constant0 = quake.custom_unitary_constant @matrix %q1
            : (!quake.wire) -> !quake.wire
        %constant1 = quake.custom_unitary_constant @matrix<adj> %constant0
            : (!quake.wire) -> !quake.wire
        %q2 = quake.null_wire
        %parameter0 = quake.custom_unitary_call @generator(%angle0) %q2
            : (f64, !quake.wire) -> !quake.wire
        %parameter1 = quake.custom_unitary_call @generator<adj>(%angle0Equal) %parameter0
            : (f64, !quake.wire) -> !quake.wire
        %parameter2 = quake.custom_unitary_call @generator(%angle1) %parameter1
            : (f64, !quake.wire) -> !quake.wire
        quake.sink %opaque : !quake.wire
        quake.sink %constant1 : !quake.wire
        quake.sink %parameter2 : !quake.wire
        return
      }
      cc.global constant private @matrix (dense<[
        (1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00),
        (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)
      ]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
    })mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "custom_unitaries");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 8u);
  CommutationAnalysis analysis(function.front());

  expectResult(analysis, operators[0], operators[1],
               commutation_status::commutes,
               commutation_reason::same_operation);
  expectResult(analysis, operators[0], operators[2],
               commutation_status::indeterminate,
               commutation_reason::no_applicable_rule);
  expectResult(analysis, operators[3], operators[4],
               commutation_status::commutes,
               commutation_reason::same_operation);
  expectResult(analysis, operators[5], operators[6],
               commutation_status::commutes,
               commutation_reason::same_operation);
  expectResult(analysis, operators[6], operators[7],
               commutation_status::indeterminate,
               commutation_reason::no_applicable_rule);
}

TEST_F(CommutationAnalysisTest, PropagatesAcrossIdentityMapGrowth) {
  constexpr unsigned identityCount = 64;
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module { func.func @growth() {\n";
  for (unsigned i = 0; i < identityCount; ++i)
    os << "  %wire" << i << " = quake.null_wire\n"
       << "  %result" << i << " = quake.x %wire" << i
       << " : (!quake.wire) -> !quake.wire\n"
       << "  quake.sink %result" << i << " : !quake.wire\n";
  os << "  return\n} }\n";

  auto module = parseModule(source);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<func::FuncOp>("growth");
  ASSERT_TRUE(function);
  auto nullWires = llvm::to_vector(function.getOps<cudaq::quake::NullWireOp>());
  auto xOps = llvm::to_vector(function.getOps<cudaq::quake::XOp>());
  ASSERT_EQ(nullWires.size(), identityCount);
  ASSERT_EQ(xOps.size(), identityCount);

  QubitIdentityAnalysis analysis(function.front());
  for (auto [nullWire, x] : llvm::zip_equal(nullWires, xOps)) {
    auto inputId = analysis.getQubitId(nullWire.getResult());
    ASSERT_TRUE(inputId);
    EXPECT_EQ(inputId, analysis.getQubitId(x.getWires().front()));
  }
}

TEST_F(CommutationAnalysisTest, PreservesKnownLanesAcrossUnknownWireOperands) {
  auto module = parseModule(R"mlir(
    module {
      func.func private @wire_source() -> !quake.wire
      func.func @partial_lanes() {
        %known = quake.null_wire
        %unknown = func.call @wire_source() : () -> !quake.wire
        %results:2 = quake.x [%unknown] %known
            : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %results#0 : !quake.wire
        quake.sink %results#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<func::FuncOp>("partial_lanes");
  ASSERT_TRUE(function);
  auto nullWire = *function.getOps<cudaq::quake::NullWireOp>().begin();
  auto x = *function.getOps<cudaq::quake::XOp>().begin();

  QubitIdentityAnalysis analysis(function.front());
  auto knownId = analysis.getQubitId(nullWire.getResult());
  ASSERT_TRUE(knownId);
  // An unidentified lane does not erase the independently known result lane.
  EXPECT_FALSE(analysis.getQubitId(x.getWires()[0]));
  EXPECT_EQ(knownId, analysis.getQubitId(x.getWires()[1]));
}

TEST_F(CommutationAnalysisTest, TracksFreshAndConservativeReferenceBoundaries) {
  auto module = parseModule(R"mlir(
    module {
      func.func @rebind(%destination: !quake.ref, %source: !quake.ref) {
        %wire = quake.unwrap %source : (!quake.ref) -> !quake.wire
        quake.wrap %wire to %destination : !quake.wire, !quake.ref
        return
      }
      func.func @fresh() {
        %first = quake.alloca !quake.ref
        %firstWire = quake.unwrap %first : (!quake.ref) -> !quake.wire
        quake.wrap %firstWire to %first : !quake.wire, !quake.ref
        %firstAfterWrap = quake.unwrap %first : (!quake.ref) -> !quake.wire
        %second = quake.alloca !quake.ref
        %secondWire = quake.unwrap %second : (!quake.ref) -> !quake.wire
        %knownWire = quake.null_wire
        %knownReference = quake.wrap_new %knownWire
            : (!quake.wire) -> !quake.ref
        %knownUnwrapped = quake.unwrap %knownReference
            : (!quake.ref) -> !quake.wire
        return
      }
      func.func @different_wrap() {
        %reference = quake.alloca !quake.ref
        %before = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        %different = quake.null_wire
        quake.wrap %different to %reference : !quake.wire, !quake.ref
        %after = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        return
      }
      func.func @untracked_wrap(%possiblyAliased: !quake.ref) {
        %first = quake.alloca !quake.ref
        %firstBefore = quake.unwrap %first : (!quake.ref) -> !quake.wire
        %second = quake.alloca !quake.ref
        %secondBefore = quake.unwrap %second : (!quake.ref) -> !quake.wire
        %wire = quake.null_wire
        quake.wrap %wire to %possiblyAliased : !quake.wire, !quake.ref
        %firstAfter = quake.unwrap %first : (!quake.ref) -> !quake.wire
        %secondAfter = quake.unwrap %second : (!quake.ref) -> !quake.wire
        return
      }
      func.func @call_boundary() {
        %destination = quake.alloca !quake.ref
        %destinationBefore = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %source = quake.alloca !quake.ref
        %sourceBefore = quake.unwrap %source : (!quake.ref) -> !quake.wire
        func.call @rebind(%destination, %source)
            : (!quake.ref, !quake.ref) -> ()
        %destinationAfter = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %sourceAfter = quake.unwrap %source : (!quake.ref) -> !quake.wire
        return
      }
      func.func @region_boundary() {
        %destination = quake.alloca !quake.ref
        %destinationBefore = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %source = quake.alloca !quake.ref
        %sourceBefore = quake.unwrap %source : (!quake.ref) -> !quake.wire
        cc.scope {
          %wire = quake.unwrap %source : (!quake.ref) -> !quake.wire
          quake.wrap %wire to %destination : !quake.wire, !quake.ref
          cc.continue
        }
        %destinationAfter = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %sourceAfter = quake.unwrap %source : (!quake.ref) -> !quake.wire
        return
      }
      func.func @cfg_boundary() {
        %wire = quake.null_wire
        cf.br ^bb1(%wire : !quake.wire)
      ^bb1(%argument: !quake.wire):
        %x = quake.x %argument : (!quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        return
      }
      func.func @effect_boundary() {
        %slot = cc.alloca i64
        %zero = arith.constant 0 : i64
        %reference = quake.alloca !quake.ref
        %before = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        cc.store %zero, %slot : !cc.ptr<i64>
        %after = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  auto fresh = module->lookupSymbol<func::FuncOp>("fresh");
  ASSERT_TRUE(fresh);
  auto freshUnwraps = llvm::to_vector(fresh.getOps<cudaq::quake::UnwrapOp>());
  auto freshNullWire = *fresh.getOps<cudaq::quake::NullWireOp>().begin();
  ASSERT_EQ(freshUnwraps.size(), 4u);
  QubitIdentityAnalysis freshAnalysis(fresh.front());
  auto firstId = freshAnalysis.getQubitId(freshUnwraps[0].getResult());
  auto secondId = freshAnalysis.getQubitId(freshUnwraps[2].getResult());
  ASSERT_TRUE(firstId);
  ASSERT_TRUE(secondId);
  EXPECT_EQ(firstId, freshAnalysis.getQubitId(freshUnwraps[1].getResult()));
  EXPECT_NE(firstId, secondId);
  EXPECT_EQ(freshAnalysis.getQubitId(freshNullWire.getResult()),
            freshAnalysis.getQubitId(freshUnwraps[3].getResult()));

  auto different = module->lookupSymbol<func::FuncOp>("different_wrap");
  ASSERT_TRUE(different);
  auto differentUnwraps =
      llvm::to_vector(different.getOps<cudaq::quake::UnwrapOp>());
  ASSERT_EQ(differentUnwraps.size(), 2u);
  QubitIdentityAnalysis differentAnalysis(different.front());
  EXPECT_TRUE(differentAnalysis.getQubitId(differentUnwraps[0].getResult()));
  EXPECT_FALSE(differentAnalysis.getQubitId(differentUnwraps[1].getResult()));

  auto untracked = module->lookupSymbol<func::FuncOp>("untracked_wrap");
  ASSERT_TRUE(untracked);
  auto untrackedUnwraps =
      llvm::to_vector(untracked.getOps<cudaq::quake::UnwrapOp>());
  ASSERT_EQ(untrackedUnwraps.size(), 4u);
  QubitIdentityAnalysis untrackedAnalysis(untracked.front());
  EXPECT_TRUE(untrackedAnalysis.getQubitId(untrackedUnwraps[0].getResult()));
  EXPECT_TRUE(untrackedAnalysis.getQubitId(untrackedUnwraps[1].getResult()));
  EXPECT_FALSE(untrackedAnalysis.getQubitId(untrackedUnwraps[2].getResult()));
  EXPECT_FALSE(untrackedAnalysis.getQubitId(untrackedUnwraps[3].getResult()));

  for (llvm::StringRef functionName : {"call_boundary", "region_boundary"}) {
    auto boundary = module->lookupSymbol<func::FuncOp>(functionName);
    ASSERT_TRUE(boundary);
    auto unwraps = llvm::to_vector(boundary.getOps<cudaq::quake::UnwrapOp>());
    ASSERT_EQ(unwraps.size(), 4u);
    QubitIdentityAnalysis analysis(boundary.front());
    EXPECT_TRUE(analysis.getQubitId(unwraps[0].getResult()));
    EXPECT_TRUE(analysis.getQubitId(unwraps[1].getResult()));
    EXPECT_FALSE(analysis.getQubitId(unwraps[2].getResult()));
    EXPECT_FALSE(analysis.getQubitId(unwraps[3].getResult()));
  }

  auto cfg = module->lookupSymbol<func::FuncOp>("cfg_boundary");
  ASSERT_TRUE(cfg);
  ASSERT_EQ(cfg.getBlocks().size(), 2u);
  Block &successor = cfg.back();
  QubitIdentityAnalysis cfgAnalysis(successor);
  EXPECT_FALSE(cfgAnalysis.getQubitId(successor.getArgument(0)));
  auto x = *successor.getOps<cudaq::quake::XOp>().begin();
  EXPECT_FALSE(cfgAnalysis.getQubitId(x.getWires().front()));

  auto effect = module->lookupSymbol<func::FuncOp>("effect_boundary");
  ASSERT_TRUE(effect);
  auto effectUnwraps = llvm::to_vector(effect.getOps<cudaq::quake::UnwrapOp>());
  ASSERT_EQ(effectUnwraps.size(), 2u);
  QubitIdentityAnalysis effectAnalysis(effect.front());
  EXPECT_TRUE(effectAnalysis.getQubitId(effectUnwraps[0].getResult()));
  EXPECT_FALSE(effectAnalysis.getQubitId(effectUnwraps[1].getResult()));
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
