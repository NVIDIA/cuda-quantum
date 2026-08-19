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
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <string>

using namespace mlir;

using QubitIdentityAnalysis = cudaq::quake::detail::QubitIdentityAnalysis;

TEST(QubitIdentityAnalysisTest, PropagatesAcrossIdentityMapGrowth) {
  constexpr unsigned identityCount = 64;
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module { func.func @growth() {\n";
  for (unsigned i = 0; i < identityCount; ++i)
    os << "  %wire" << i << " = quake.null_wire\n"
       << "  %result" << i << " = quake.x %wire" << i
       << " : (!quake.wire) -> !quake.wire\n"
       << "  quake.sink %result" << i << " : !quake.wire\n";
  os << "  return\n"
        "} }\n";

  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();
  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto function = module->lookupSymbol<func::FuncOp>("growth");
  ASSERT_TRUE(function);

  auto nullWires = llvm::to_vector(function.getOps<cudaq::quake::NullWireOp>());
  auto xOps = llvm::to_vector(function.getOps<cudaq::quake::XOp>());
  ASSERT_EQ(nullWires.size(), identityCount);
  ASSERT_EQ(xOps.size(), identityCount);

  // Repeated source/result pairs grow the identity map several times while
  // scalar-wire results inherit the source identity.
  QubitIdentityAnalysis analysis(function.front());
  for (auto [nullWire, x] : llvm::zip_equal(nullWires, xOps)) {
    auto inputId = analysis.getQubitId(nullWire.getResult());
    ASSERT_TRUE(inputId);
    EXPECT_EQ(inputId, analysis.getQubitId(x.getWires().front()));
  }
}

TEST(QubitIdentityAnalysisTest, DistinguishesKnownDisjointAndAliasedWires) {
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
                          %aggregate: !quake.veq<2>,
                          %referenceArg: !quake.ref) {
        %initial = quake.null_wire
        %x = quake.x %initial : (!quake.wire) -> !quake.wire
        %reset = quake.reset %x : (!quake.wire) -> !quake.wire
        %distinct = quake.null_wire
        %measurementInput = quake.null_wire
        %measurement, %measuredInitial, %measuredDistinct =
            quake.mz %measurementInput, %distinct
                : (!quake.wire, !quake.wire)
                  -> (!cc.sequence<!cc.measure_handle>, !quake.wire, !quake.wire)
        %conversionInput = quake.null_wire
        %control = quake.to_ctrl %conversionInput
            : (!quake.wire) -> !quake.control
        %returned = quake.from_ctrl %control
            : (!quake.control) -> !quake.wire
        %wireControl = quake.null_wire
        %wireTarget = quake.null_wire
        %wireResults:2 = quake.x [%wireControl] %wireTarget
            : (!quake.wire, !quake.wire)
              -> (!quake.wire, !quake.wire)
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
        %unwrapped = quake.unwrap %referenceArg : (!quake.ref) -> !quake.wire
        quake.sink %returned : !quake.wire
        quake.sink %reset : !quake.wire
        quake.sink %measuredInitial : !quake.wire
        quake.sink %measuredDistinct : !quake.wire
        quake.sink %wireResults#0 : !quake.wire
        quake.sink %wireResults#1 : !quake.wire
        quake.sink %mixedResults#0 : !quake.wire
        quake.sink %mixedResults#1 : !quake.wire
        quake.sink %call : !quake.wire
        quake.sink %wireArg : !quake.wire
        %controlArgumentWire = quake.from_ctrl %controlArg
            : (!quake.control) -> !quake.wire
        quake.sink %controlArgumentWire : !quake.wire
        quake.wrap %unwrapped to %referenceArg : !quake.wire, !quake.ref
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
  ASSERT_EQ(nullWires.size(), 9u);
  ASSERT_EQ(xOps.size(), 3u);
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
  auto wireX = xOps[1];
  auto mixedX = xOps[2];
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
  // Operators thread identities through scalar wire controls and targets.
  EXPECT_EQ(initialId, analysis.getQubitId(x.getWires().front()));
  ASSERT_EQ(wireX.getWires().size(), 2u);
  EXPECT_EQ(analysis.getQubitId(wireX.getControls()[0]),
            analysis.getQubitId(wireX.getWires()[0]));
  EXPECT_EQ(analysis.getQubitId(wireX.getTargets()[0]),
            analysis.getQubitId(wireX.getWires()[1]));

  // Any reusable control makes the whole operator unsupported for identity
  // propagation, even when its other operands and results are scalar wires.
  ASSERT_EQ(mixedX.getWires().size(), 2u);
  EXPECT_FALSE(analysis.getQubitId(mixedX.getWires()[0]));
  EXPECT_FALSE(analysis.getQubitId(mixedX.getWires()[1]));

  // Block arguments have no identity without a verified no-alias guarantee.
  EXPECT_FALSE(analysis.getQubitId(function.getArgument(0)));
  EXPECT_FALSE(analysis.getQubitId(function.getArgument(1)));
  EXPECT_FALSE(analysis.getQubitId(function.getArgument(3)));
  ASSERT_TRUE(analysis.getQubitId(distinct));
  EXPECT_NE(initialId, analysis.getQubitId(distinct));

  // A returned and reborrowed wire retains its wire-set identity, while a
  // different wire-set index identifies a different virtual qubit.
  EXPECT_EQ(analysis.getQubitId(borrow0a), analysis.getQubitId(borrow0b));
  EXPECT_NE(analysis.getQubitId(borrow0a), analysis.getQubitId(borrow1));

  // Reset and measurement preserve each scalar wire's block-local identity.
  EXPECT_EQ(analysis.getQubitId(x.getWires().front()),
            analysis.getQubitId(reset.getWires().front()));
  EXPECT_EQ(analysis.getQubitId(measurement.getTargets()[0]),
            analysis.getQubitId(measurement.getWires()[0]));
  EXPECT_EQ(analysis.getQubitId(measurement.getTargets()[1]),
            analysis.getQubitId(measurement.getWires()[1]));

  // Conversions, aggregates, calls, and references remain unsupported
  // boundaries.
  EXPECT_FALSE(analysis.getQubitId(control));
  EXPECT_FALSE(analysis.getQubitId(returned));
  EXPECT_FALSE(analysis.getQubitId(controlArgumentWire));
  EXPECT_FALSE(analysis.getQubitId(function.getArgument(2)));
  EXPECT_FALSE(analysis.getQubitId(call.getResult(0)));
  EXPECT_FALSE(analysis.getQubitId(unwrapped.getResult()));
}

TEST(QubitIdentityAnalysisTest, LeavesSuccessorArgumentsUnknown) {
  MLIRContext context;
  context.loadDialect<cf::ControlFlowDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
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
    }
  )mlir",
                                            &context);

  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto function = module->lookupSymbol<func::FuncOp>("successor_arguments");
  ASSERT_TRUE(function);
  ASSERT_EQ(function.getBlocks().size(), 2u);

  Block &successor = function.back();
  QubitIdentityAnalysis analysis(successor);
  EXPECT_FALSE(analysis.getQubitId(successor.getArgument(0)));
  EXPECT_FALSE(analysis.getQubitId(successor.getArgument(1)));
  auto operators =
      llvm::to_vector(successor.getOps<cudaq::quake::OperatorInterface>());
  ASSERT_EQ(operators.size(), 2u);
  EXPECT_FALSE(analysis.getQubitId(operators[0].getTargets().front()));
  EXPECT_FALSE(analysis.getQubitId(operators[1].getTargets().front()));
}

TEST(QubitIdentityAnalysisTest, TracksProvenFreshReferenceProvenance) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::cc::CCDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func private @reference_source() -> !quake.ref
      func.func @rebind(%destination: !quake.ref, %source: !quake.ref) {
        %wire = quake.unwrap %source : (!quake.ref) -> !quake.wire
        quake.wrap %wire to %destination : !quake.wire, !quake.ref
        return
      }
      func.func @rebind_closure(%callable: !cc.callable<() -> ()>) {
        %references:2 = cc.callable_closure %callable
            : (!cc.callable<() -> ()>) -> (!quake.ref, !quake.ref)
        %wire = quake.unwrap %references#1 : (!quake.ref) -> !quake.wire
        quake.wrap %wire to %references#0 : !quake.wire, !quake.ref
        return
      }
      func.func private @empty_action()

      func.func @fresh_references(%unknownWire: !quake.wire) {
        %first = quake.alloca !quake.ref
        %firstInitial = quake.unwrap %first : (!quake.ref) -> !quake.wire
        quake.wrap %firstInitial to %first : !quake.wire, !quake.ref
        %firstAfterWrap = quake.unwrap %first : (!quake.ref) -> !quake.wire
        %firstRepeated = quake.unwrap %first : (!quake.ref) -> !quake.wire

        %second = quake.alloca !quake.ref
        %secondInitial = quake.unwrap %second : (!quake.ref) -> !quake.wire

        %knownWire = quake.null_wire
        %knownReference = quake.wrap_new %knownWire
            : (!quake.wire) -> !quake.ref
        %knownUnwrapped = quake.unwrap %knownReference
            : (!quake.ref) -> !quake.wire

        %unknownReference = quake.wrap_new %unknownWire
            : (!quake.wire) -> !quake.ref
        %unknownUnwrapped = quake.unwrap %unknownReference
            : (!quake.ref) -> !quake.wire
        return
      }

      func.func @uncertain_references(%condition: i1,
                                      %referenceArg: !quake.ref,
                                      %vectorArg: !quake.veq<2>) {
        %local = quake.alloca !quake.ref
        %unrelated = cc.undef !quake.ref
        %cast = builtin.unrealized_conversion_cast %local
            : !quake.ref to !quake.ref
        %selected = arith.select %condition, %local, %local : !quake.ref
        %aggregate = quake.make_struq %local, %referenceArg
            : (!quake.ref, !quake.ref)
              -> !quake.struq<!quake.ref, !quake.ref>
        %member = quake.get_member %aggregate[0]
            : (!quake.struq<!quake.ref, !quake.ref>) -> !quake.ref
        %concatenated = quake.concat %local, %referenceArg
            : (!quake.ref, !quake.ref) -> !quake.veq<2>
        %fromConcat = quake.extract_ref %concatenated[0]
            : (!quake.veq<2>) -> !quake.ref
        %fromVector = quake.extract_ref %vectorArg[0]
            : (!quake.veq<2>) -> !quake.ref

        %localAfterPureConstructions = quake.unwrap %local
            : (!quake.ref) -> !quake.wire
        %argWire = quake.unwrap %referenceArg
            : (!quake.ref) -> !quake.wire
        %unrelatedWire = quake.unwrap %unrelated
            : (!quake.ref) -> !quake.wire
        %castWire = quake.unwrap %cast : (!quake.ref) -> !quake.wire
        %selectedWire = quake.unwrap %selected
            : (!quake.ref) -> !quake.wire
        %memberWire = quake.unwrap %member : (!quake.ref) -> !quake.wire
        %concatWire = quake.unwrap %fromConcat
            : (!quake.ref) -> !quake.wire
        %vectorWire = quake.unwrap %fromVector
            : (!quake.ref) -> !quake.wire
        %call = func.call @reference_source() : () -> !quake.ref
        %callWire = quake.unwrap %call : (!quake.ref) -> !quake.wire
        return
      }

      func.func @different_direct_wrap() {
        %reference = quake.alloca !quake.ref
        %before = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        %different = quake.null_wire
        quake.wrap %different to %reference : !quake.wire, !quake.ref
        %after = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        return
      }

      func.func @unknown_direct_wrap(%unknownWire: !quake.wire) {
        %reference = quake.alloca !quake.ref
        %before = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        quake.wrap %unknownWire to %reference : !quake.wire, !quake.ref
        %after = quake.unwrap %reference : (!quake.ref) -> !quake.wire
        return
      }

      func.func @aliased_wrap() {
        %first = quake.alloca !quake.ref
        %firstBefore = quake.unwrap %first : (!quake.ref) -> !quake.wire
        %second = quake.alloca !quake.ref
        %secondWire = quake.unwrap %second : (!quake.ref) -> !quake.wire
        %aliases = quake.concat %first
            : (!quake.ref) -> !quake.veq<1>
        %alias = quake.extract_ref %aliases[0]
            : (!quake.veq<1>) -> !quake.ref
        %firstAfterAliasConstruction = quake.unwrap %first
            : (!quake.ref) -> !quake.wire
        quake.wrap %secondWire to %alias : !quake.wire, !quake.ref
        %firstAfter = quake.unwrap %first : (!quake.ref) -> !quake.wire
        return
      }

      func.func @call_boundary() {
        %destination = quake.alloca !quake.ref
        %destinationBefore = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %source = quake.alloca !quake.ref
        %sourceBefore = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        func.call @rebind(%destination, %source)
            : (!quake.ref, !quake.ref) -> ()
        %destinationAfter = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %sourceAfter = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        return
      }

      func.func @region_boundary() {
        %destination = quake.alloca !quake.ref
        %destinationBefore = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %source = quake.alloca !quake.ref
        %sourceBefore = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        cc.scope {
          %wire = quake.unwrap %source : (!quake.ref) -> !quake.wire
          quake.wrap %wire to %destination : !quake.wire, !quake.ref
          cc.continue
        }
        %destinationAfter = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %sourceAfter = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        return
      }

      func.func @compute_action_boundary() {
        %destination = quake.alloca !quake.ref
        %destinationBefore = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %source = quake.alloca !quake.ref
        %sourceBefore = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        %compute = cc.instantiate_callable @rebind_closure(%destination, %source)
            : (!quake.ref, !quake.ref) -> !cc.callable<() -> ()>
        %action = cc.instantiate_callable @empty_action() nocapture
            : () -> !cc.callable<() -> ()>
        quake.compute_action %compute, %action
            : !cc.callable<() -> ()>, !cc.callable<() -> ()>
        %destinationAfter = quake.unwrap %destination
            : (!quake.ref) -> !quake.wire
        %sourceAfter = quake.unwrap %source
            : (!quake.ref) -> !quake.wire
        return
      }
    }
  )mlir",
                                            &context);

  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  auto fresh = module->lookupSymbol<func::FuncOp>("fresh_references");
  ASSERT_TRUE(fresh);
  auto freshUnwraps = llvm::to_vector(fresh.getOps<cudaq::quake::UnwrapOp>());
  auto nullWires = llvm::to_vector(fresh.getOps<cudaq::quake::NullWireOp>());
  ASSERT_EQ(freshUnwraps.size(), 6u);
  ASSERT_EQ(nullWires.size(), 1u);

  QubitIdentityAnalysis freshAnalysis(fresh.front());
  auto firstId = freshAnalysis.getQubitId(freshUnwraps[0].getResult());
  ASSERT_TRUE(firstId);
  EXPECT_EQ(firstId, freshAnalysis.getQubitId(freshUnwraps[1].getResult()));
  EXPECT_EQ(firstId, freshAnalysis.getQubitId(freshUnwraps[2].getResult()));
  auto secondId = freshAnalysis.getQubitId(freshUnwraps[3].getResult());
  ASSERT_TRUE(secondId);
  EXPECT_NE(firstId, secondId);
  EXPECT_EQ(freshAnalysis.getQubitId(nullWires[0].getResult()),
            freshAnalysis.getQubitId(freshUnwraps[4].getResult()));
  EXPECT_FALSE(freshAnalysis.getQubitId(fresh.getArgument(0)));
  EXPECT_FALSE(freshAnalysis.getQubitId(freshUnwraps[5].getResult()));

  auto uncertain = module->lookupSymbol<func::FuncOp>("uncertain_references");
  ASSERT_TRUE(uncertain);
  auto uncertainUnwraps =
      llvm::to_vector(uncertain.getOps<cudaq::quake::UnwrapOp>());
  ASSERT_EQ(uncertainUnwraps.size(), 9u);

  QubitIdentityAnalysis uncertainAnalysis(uncertain.front());
  EXPECT_FALSE(uncertainAnalysis.getQubitId(uncertain.getArgument(1)));
  EXPECT_TRUE(
      uncertainAnalysis.getQubitId(uncertainUnwraps.front().getResult()));
  for (auto unwrap : llvm::drop_begin(uncertainUnwraps))
    EXPECT_FALSE(uncertainAnalysis.getQubitId(unwrap.getResult()));

  auto different = module->lookupSymbol<func::FuncOp>("different_direct_wrap");
  ASSERT_TRUE(different);
  auto differentUnwraps =
      llvm::to_vector(different.getOps<cudaq::quake::UnwrapOp>());
  ASSERT_EQ(differentUnwraps.size(), 2u);
  QubitIdentityAnalysis differentAnalysis(different.front());
  EXPECT_TRUE(differentAnalysis.getQubitId(differentUnwraps[0].getResult()));
  EXPECT_FALSE(differentAnalysis.getQubitId(differentUnwraps[1].getResult()));

  auto unknown = module->lookupSymbol<func::FuncOp>("unknown_direct_wrap");
  ASSERT_TRUE(unknown);
  auto unknownUnwraps =
      llvm::to_vector(unknown.getOps<cudaq::quake::UnwrapOp>());
  ASSERT_EQ(unknownUnwraps.size(), 2u);
  QubitIdentityAnalysis unknownAnalysis(unknown.front());
  EXPECT_TRUE(unknownAnalysis.getQubitId(unknownUnwraps[0].getResult()));
  EXPECT_FALSE(unknownAnalysis.getQubitId(unknownUnwraps[1].getResult()));

  auto aliased = module->lookupSymbol<func::FuncOp>("aliased_wrap");
  ASSERT_TRUE(aliased);
  auto aliasedUnwraps =
      llvm::to_vector(aliased.getOps<cudaq::quake::UnwrapOp>());
  ASSERT_EQ(aliasedUnwraps.size(), 4u);
  QubitIdentityAnalysis aliasedAnalysis(aliased.front());
  auto firstBefore = aliasedAnalysis.getQubitId(aliasedUnwraps[0].getResult());
  auto second = aliasedAnalysis.getQubitId(aliasedUnwraps[1].getResult());
  ASSERT_TRUE(firstBefore);
  ASSERT_TRUE(second);
  EXPECT_NE(firstBefore, second);
  EXPECT_EQ(firstBefore,
            aliasedAnalysis.getQubitId(aliasedUnwraps[2].getResult()));
  EXPECT_FALSE(aliasedAnalysis.getQubitId(aliasedUnwraps[3].getResult()));

  for (llvm::StringRef functionName :
       {"call_boundary", "region_boundary", "compute_action_boundary"}) {
    auto boundary = module->lookupSymbol<func::FuncOp>(functionName);
    ASSERT_TRUE(boundary);
    auto boundaryUnwraps =
        llvm::to_vector(boundary.getOps<cudaq::quake::UnwrapOp>());
    ASSERT_EQ(boundaryUnwraps.size(), 4u);
    QubitIdentityAnalysis boundaryAnalysis(boundary.front());
    auto destinationBefore =
        boundaryAnalysis.getQubitId(boundaryUnwraps[0].getResult());
    auto sourceBefore =
        boundaryAnalysis.getQubitId(boundaryUnwraps[1].getResult());
    ASSERT_TRUE(destinationBefore);
    ASSERT_TRUE(sourceBefore);
    EXPECT_NE(destinationBefore, sourceBefore);
    EXPECT_FALSE(boundaryAnalysis.getQubitId(boundaryUnwraps[2].getResult()));
    EXPECT_FALSE(boundaryAnalysis.getQubitId(boundaryUnwraps[3].getResult()));
  }
}
