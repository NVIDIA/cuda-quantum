/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QubitIdentityAnalysis.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;

using cudaq::quake::detail::QubitIdentityAnalysis;

TEST(QubitIdentityAnalysisTest, TracksQubitIdentity) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::cc::CCDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      quake.wire_set @wires[2]
      func.func private @wire_source() -> !quake.wire
      func.func @identity(%wireArg: !quake.wire,
                          %controlArg: !quake.control,
                          %aggregate: !quake.veq<2>) {
        %initial = quake.null_wire
        %x = quake.x %initial : (!quake.wire) -> !quake.wire
        %reset = quake.reset %x : (!quake.wire) -> !quake.wire
        %distinct = quake.null_wire
        %measurement, %measuredInitial, %measuredDistinct =
            quake.mz %reset, %distinct
                : (!quake.wire, !quake.wire)
                  -> (!cc.stdvec<!quake.measure>, !quake.wire, !quake.wire)
        %control = quake.to_ctrl %measuredInitial
            : (!quake.wire) -> !quake.control
        %returned = quake.from_ctrl %control
            : (!quake.control) -> !quake.wire
        %mixedWireControl = quake.null_wire
        %mixedControlWire = quake.null_wire
        %mixedTarget = quake.null_wire
        %mixedControl = quake.to_ctrl %mixedControlWire
            : (!quake.wire) -> !quake.control
        %mixedResults:2 = quake.x [%mixedWireControl, %mixedControl] %mixedTarget
            : (!quake.wire, !quake.control, !quake.wire)
              -> (!quake.wire, !quake.wire)

        %borrow0a = quake.borrow_wire @wires[0] : !quake.wire
        quake.return_wire %borrow0a : !quake.wire
        %borrow0b = quake.borrow_wire @wires[0] : !quake.wire
        quake.return_wire %borrow0b : !quake.wire
        %borrow1 = quake.borrow_wire @wires[1] : !quake.wire
        quake.return_wire %borrow1 : !quake.wire

        %call = func.call @wire_source() : () -> !quake.wire
        %reference = quake.alloca !quake.ref
        %unwrapped = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        quake.sink %returned : !quake.wire
        quake.sink %measuredDistinct : !quake.wire
        quake.sink %mixedResults#0 : !quake.wire
        quake.sink %mixedResults#1 : !quake.wire
        quake.sink %call : !quake.wire
        quake.sink %wireArg : !quake.wire
        %controlArgumentWire = quake.from_ctrl %controlArg
            : (!quake.control) -> !quake.wire
        quake.sink %controlArgumentWire : !quake.wire
        quake.wrap %unwrapped to %reference : !quake.wire, !quake.ref
        return
      }
    }
  )mlir",
                                            &context);

  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto function = module->lookupSymbol<func::FuncOp>("identity");
  ASSERT_TRUE(function);

  Block &block = function.front();
  auto nullWires = llvm::to_vector(block.getOps<cudaq::quake::NullWireOp>());
  auto xOps = llvm::to_vector(block.getOps<cudaq::quake::XOp>());
  auto resets = llvm::to_vector(block.getOps<cudaq::quake::ResetOp>());
  auto measurements = llvm::to_vector(block.getOps<cudaq::quake::MzOp>());
  auto toControls = llvm::to_vector(block.getOps<cudaq::quake::ToControlOp>());
  auto fromControls =
      llvm::to_vector(block.getOps<cudaq::quake::FromControlOp>());
  auto borrows = llvm::to_vector(block.getOps<cudaq::quake::BorrowWireOp>());
  auto calls = llvm::to_vector(block.getOps<func::CallOp>());
  auto unwraps = llvm::to_vector(block.getOps<cudaq::quake::UnwrapOp>());
  ASSERT_EQ(nullWires.size(), 5u);
  ASSERT_EQ(xOps.size(), 2u);
  ASSERT_EQ(resets.size(), 1u);
  ASSERT_EQ(measurements.size(), 1u);
  ASSERT_EQ(toControls.size(), 2u);
  ASSERT_EQ(fromControls.size(), 2u);
  ASSERT_EQ(borrows.size(), 3u);
  ASSERT_EQ(calls.size(), 1u);
  ASSERT_EQ(unwraps.size(), 1u);

  Value initial = nullWires[0].getResult();
  Value distinct = nullWires[1].getResult();
  auto x = xOps[0];
  auto mixedX = xOps[1];
  auto reset = resets[0];
  auto measurement = measurements[0];
  auto control = toControls[0];
  Value returned = fromControls[0].getResult();
  Value controlArgumentWire = fromControls[1].getResult();
  Value borrow0a = borrows[0].getResult();
  Value borrow0b = borrows[1].getResult();
  Value borrow1 = borrows[2].getResult();
  auto call = calls[0];
  auto unwrapped = unwraps[0];

  QubitIdentityAnalysis analysis(function.front());
  auto initialId = analysis.getQubitId(initial);
  ASSERT_TRUE(initialId);
  // State-changing and control-conversion operations preserve virtual-qubit
  // identity as they replace their scalar SSA inputs.
  EXPECT_EQ(initialId, analysis.getQubitId(x.getWires().front()));
  EXPECT_EQ(initialId, analysis.getQubitId(reset.getWires().front()));
  EXPECT_EQ(initialId, analysis.getQubitId(measurement.getWires().front()));
  EXPECT_EQ(initialId, analysis.getQubitId(control));
  EXPECT_EQ(initialId, analysis.getQubitId(returned));

  // Operator wire results follow wire controls and targets in operand order;
  // !quake.control operands do not produce replacement wire results.
  ASSERT_EQ(mixedX.getWires().size(), 2u);
  ASSERT_TRUE(analysis.getQubitId(mixedX.getWires()[0]));
  ASSERT_TRUE(analysis.getQubitId(mixedX.getWires()[1]));
  EXPECT_EQ(analysis.getQubitId(mixedX.getControls()[0]),
            analysis.getQubitId(mixedX.getWires()[0]));
  EXPECT_EQ(analysis.getQubitId(mixedX.getTargets()[0]),
            analysis.getQubitId(mixedX.getWires()[1]));

  // Scalar block arguments and null wires introduce distinct local identities.
  ASSERT_TRUE(analysis.getQubitId(function.getArgument(0)));
  ASSERT_TRUE(analysis.getQubitId(function.getArgument(1)));
  EXPECT_EQ(analysis.getQubitId(function.getArgument(1)),
            analysis.getQubitId(controlArgumentWire));
  ASSERT_TRUE(analysis.getQubitId(distinct));
  EXPECT_NE(initialId, analysis.getQubitId(distinct));
  EXPECT_EQ(analysis.getQubitId(distinct),
            analysis.getQubitId(measurement.getWires()[1]));

  // A returned and reborrowed wire retains its wire-set identity, while a
  // different wire-set index identifies a different virtual qubit.
  EXPECT_EQ(analysis.getQubitId(borrow0a), analysis.getQubitId(borrow0b));
  EXPECT_NE(analysis.getQubitId(borrow0a), analysis.getQubitId(borrow1));

  // Identity propagation through aggregates, calls, and references is
  // deliberately unsupported.
  EXPECT_FALSE(analysis.getQubitId(function.getArgument(2)));
  EXPECT_FALSE(analysis.getQubitId(call.getResult(0)));
  EXPECT_FALSE(analysis.getQubitId(unwrapped.getResult()));
}
