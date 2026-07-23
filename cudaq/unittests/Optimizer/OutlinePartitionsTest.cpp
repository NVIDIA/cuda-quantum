/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir { class Operation; }
namespace cudaq::opt {
mlir::FailureOr<cudaq::cc::CreateLambdaOp>
outlinePartition(llvm::ArrayRef<mlir::Operation *>);
mlir::LogicalResult
outlinePartitions(llvm::ArrayRef<llvm::SmallVector<mlir::Operation *>>);
} // namespace cudaq::opt
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <gtest/gtest.h>

using namespace mlir;

static OwningOpRef<ModuleOp> parse(MLIRContext &ctx, const char *ir) {
  ctx.loadDialect<arith::ArithDialect, func::FuncDialect,
                  cudaq::cc::CCDialect, cudaq::quake::QuakeDialect>();
  return parseSourceString<ModuleOp>(ir, &ctx);
}


// A contiguous run of two gates is outlined into a closure taking the two input
// wires and returning the two output wires; the original gates are removed.
TEST(OutlinePartitions, ContiguousRun) {
  const char *ir = R"(
    func.func @k() {
      %w0 = quake.null_wire
      %w1 = quake.null_wire
      %h = quake.h %w0 : (!quake.wire) -> !quake.wire
      %x:2 = quake.x [%h] %w1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      return
    }
  )";
  MLIRContext ctx;
  auto mod = parse(ctx, ir);
  ASSERT_TRUE(mod);

  SmallVector<Operation *> partition;
  mod->walk([&](Operation *op) {
    if (isa<cudaq::quake::HOp, cudaq::quake::XOp>(op))
      partition.push_back(op);
  });
  ASSERT_EQ(partition.size(), 2u);

  auto lambda = cudaq::opt::outlinePartition(partition);
  ASSERT_TRUE(succeeded(lambda));
  EXPECT_TRUE(succeeded(verify(*mod)));

  // The closure's signature: two wires in, two wires out.
  auto sig = lambda->getType().getSignature();
  EXPECT_EQ(sig.getNumInputs(), 2u);
  EXPECT_EQ(sig.getNumResults(), 2u);

  // Exactly one create_lambda and one call_callable; the gates moved into the
  // closure body, so none remain in the enclosing function's block.
  unsigned lambdas = 0, calls = 0, strayGates = 0;
  mod->walk([&](Operation *op) {
    if (isa<cudaq::cc::CreateLambdaOp>(op))
      ++lambdas;
    else if (isa<cudaq::cc::CallCallableOp>(op))
      ++calls;
  });
  for (Operation &op : lambda->getOperation()->getBlock()->getOperations())
    if (isa<cudaq::quake::HOp, cudaq::quake::XOp>(op))
      ++strayGates;
  EXPECT_EQ(lambdas, 1u);
  EXPECT_EQ(calls, 1u);
  EXPECT_EQ(strayGates, 0u);
}

// A partition whose output wire is consumed by a non-partition op before the
// block ends is still valid: the closure is inserted before the consumer so
// its results dominate all uses.
TEST(OutlinePartitions, EarlyConsumerReordered) {
  const char *ir = R"(
    func.func @k() {
      %w0 = quake.null_wire
      %w1 = quake.null_wire
      %h = quake.h %w0 : (!quake.wire) -> !quake.wire
      %z = quake.z %h : (!quake.wire) -> !quake.wire
      %t = quake.t %w1 : (!quake.wire) -> !quake.wire
      return
    }
  )";
  MLIRContext ctx;
  auto mod = parse(ctx, ir);
  ASSERT_TRUE(mod);

  // Partition {h, t}: h's output feeds z (non-partition) which appears before
  // t in block order, but there is no non-partition op on a wire path between
  // h and t, so the partition is contiguous. The closure is inserted before z.
  SmallVector<Operation *> partition;
  mod->walk([&](Operation *op) {
    if (isa<cudaq::quake::HOp, cudaq::quake::TOp>(op))
      partition.push_back(op);
  });
  ASSERT_EQ(partition.size(), 2u);
  auto lambda = cudaq::opt::outlinePartition(partition);
  ASSERT_TRUE(succeeded(lambda));
  EXPECT_TRUE(succeeded(verify(*mod)));
}

// A non-partition op that lies on a wire path between two partition ops is a
// true contiguity violation and must be rejected.
TEST(OutlinePartitions, NonContiguousRejected) {
  const char *ir = R"(
    func.func @k() {
      %w0 = quake.null_wire
      %h = quake.h %w0 : (!quake.wire) -> !quake.wire
      %z = quake.z %h : (!quake.wire) -> !quake.wire
      %x = quake.x %z : (!quake.wire) -> !quake.wire
      return
    }
  )";
  MLIRContext ctx;
  auto mod = parse(ctx, ir);
  ASSERT_TRUE(mod);

  // Partition {h, x}: the wire path h -> z (non-partition) -> x passes through
  // a non-partition op, so the partition is not a contiguous slice.
  SmallVector<Operation *> partition;
  mod->walk([&](Operation *op) {
    if (isa<cudaq::quake::HOp, cudaq::quake::XOp>(op))
      partition.push_back(op);
  });
  ASSERT_EQ(partition.size(), 2u);
  EXPECT_TRUE(failed(cudaq::opt::outlinePartition(partition)));
}

// Run the pass with a hardcoded analysis that puts H and X into one partition,
// verifying the full analysis → pass → outline pipeline.
TEST(OutlinePartitions, HardcodedAnalysis) {
  const char *ir = R"(
    func.func @k() {
      %w0 = quake.null_wire
      %w1 = quake.null_wire
      %h = quake.h %w0 : (!quake.wire) -> !quake.wire
      %x:2 = quake.x [%h] %w1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      return
    }
  )";
  MLIRContext ctx;
  auto mod = parse(ctx, ir);
  ASSERT_TRUE(mod);

  struct HardcodedAnalysis {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HardcodedAnalysis)
    explicit HardcodedAnalysis(mlir::Operation *op) {
      SmallVector<Operation *> partition;
      op->walk([&](Operation *inner) {
        if (isa<cudaq::quake::HOp, cudaq::quake::XOp>(inner))
          partition.push_back(inner);
      });
      if (!partition.empty())
        partitions.push_back(std::move(partition));
    }
    ArrayRef<SmallVector<Operation *>> getPartitions() const { return partitions; }
  private:
    SmallVector<SmallVector<Operation *>> partitions;
  };
  func::FuncOp funcOp;
  mod->walk([&](func::FuncOp f) { funcOp = f; });
  ASSERT_TRUE(funcOp);

  HardcodedAnalysis analysis(funcOp.getOperation());
  ASSERT_TRUE(succeeded(cudaq::opt::outlinePartitions(analysis.getPartitions())));
  EXPECT_TRUE(succeeded(verify(*mod)));

  unsigned lambdas = 0, calls = 0;
  mod->walk([&](Operation *op) {
    if (isa<cudaq::cc::CreateLambdaOp>(op)) ++lambdas;
    else if (isa<cudaq::cc::CallCallableOp>(op)) ++calls;
  });
  EXPECT_EQ(lambdas, 1u);
  EXPECT_EQ(calls, 1u);
}

// Three partitions where outputs from partitions 1 and 2 feed into partition 3:
//   P1 = {H}  on w0          → produces h_out
//   P2 = {T}  on w1          → produces t_out
//   P3 = {X}  [h_out] t_out  → consumes both h_out and t_out
// The outlined IR should chain three lambdas, with the call results of
// lambdas 1 and 2 flowing as inputs to lambda 3.
TEST(OutlinePartitions, ThreePartitionsWithCrossFlow) {
  const char *ir = R"(
    func.func @k() {
      %w0 = quake.null_wire
      %w1 = quake.null_wire
      %h = quake.h %w0 : (!quake.wire) -> !quake.wire
      %t = quake.t %w1 : (!quake.wire) -> !quake.wire
      %x:2 = quake.x [%h] %t : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      return
    }
  )";
  MLIRContext ctx;
  auto mod = parse(ctx, ir);
  ASSERT_TRUE(mod);

  struct ThreePartitionAnalysis {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreePartitionAnalysis)
    explicit ThreePartitionAnalysis(mlir::Operation *op) {
      SmallVector<Operation *> p1, p2, p3;
      op->walk([&](Operation *inner) {
        if (isa<cudaq::quake::HOp>(inner))      p1.push_back(inner);
        else if (isa<cudaq::quake::TOp>(inner)) p2.push_back(inner);
        else if (isa<cudaq::quake::XOp>(inner)) p3.push_back(inner);
      });
      if (!p1.empty()) partitions.push_back(std::move(p1));
      if (!p2.empty()) partitions.push_back(std::move(p2));
      if (!p3.empty()) partitions.push_back(std::move(p3));
    }
    ArrayRef<SmallVector<Operation *>> getPartitions() const { return partitions; }
  private:
    SmallVector<SmallVector<Operation *>> partitions;
  };

  func::FuncOp funcOp;
  mod->walk([&](func::FuncOp f) { funcOp = f; });
  ASSERT_TRUE(funcOp);

  ThreePartitionAnalysis analysis(funcOp.getOperation());
  ASSERT_TRUE(succeeded(cudaq::opt::outlinePartitions(analysis.getPartitions())));
  ASSERT_TRUE(succeeded(verify(*mod)));

  // Collect lambdas and calls in block order.
  SmallVector<cudaq::cc::CreateLambdaOp> lambdas;
  SmallVector<cudaq::cc::CallCallableOp> calls;
  funcOp.walk([&](cudaq::cc::CreateLambdaOp l) { lambdas.push_back(l); });
  funcOp.walk([&](cudaq::cc::CallCallableOp c) { calls.push_back(c); });
  ASSERT_EQ(lambdas.size(), 3u);
  ASSERT_EQ(calls.size(), 3u);

  // P1: lambda body contains H, single-wire signature.
  unsigned hOps = 0;
  lambdas[0].walk([&](cudaq::quake::HOp) { ++hOps; });
  EXPECT_EQ(hOps, 1u);
  EXPECT_EQ(lambdas[0].getType().getSignature().getNumInputs(), 1u);
  EXPECT_EQ(lambdas[0].getType().getSignature().getNumResults(), 1u);

  // P2: lambda body contains T, single-wire signature.
  unsigned tOps = 0;
  lambdas[1].walk([&](cudaq::quake::TOp) { ++tOps; });
  EXPECT_EQ(tOps, 1u);
  EXPECT_EQ(lambdas[1].getType().getSignature().getNumInputs(), 1u);
  EXPECT_EQ(lambdas[1].getType().getSignature().getNumResults(), 1u);

  // P3: lambda body contains X, two-wire signature.
  unsigned xOps = 0;
  lambdas[2].walk([&](cudaq::quake::XOp) { ++xOps; });
  EXPECT_EQ(xOps, 1u);
  EXPECT_EQ(lambdas[2].getType().getSignature().getNumInputs(), 2u);
  EXPECT_EQ(lambdas[2].getType().getSignature().getNumResults(), 2u);

  // Cross-flow: the two wire arguments to P3's call are the results of P1
  // and P2's calls — not raw null_wire values.
  // calls[2] operands: [callable, wire0, wire1]
  EXPECT_EQ(calls[2].getOperand(1), calls[0].getResult(0));
  EXPECT_EQ(calls[2].getOperand(2), calls[1].getResult(0));
}
