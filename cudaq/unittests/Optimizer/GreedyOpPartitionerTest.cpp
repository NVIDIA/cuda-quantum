/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/GreedyOpPartitioner.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include <gtest/gtest.h>

using namespace mlir;

static OwningOpRef<ModuleOp> parse(MLIRContext &ctx, const char *ir) {
  ctx.loadDialect<func::FuncDialect, cudaq::cc::CCDialect,
                  cudaq::quake::QuakeDialect>();
  return parseSourceString<ModuleOp>(ir, &ctx);
}

static func::FuncOp getFunc(ModuleOp mod) {
  func::FuncOp result;
  mod.walk([&](func::FuncOp f) { result = f; });
  return result;
}

// Returns the index of the partition that contains op, or -1 if none.
static int partitionOf(ArrayRef<SmallVector<Operation *>> partitions,
                       Operation *op) {
  for (auto [i, part] : llvm::enumerate(partitions))
    for (Operation *o : part)
      if (o == op)
        return static_cast<int>(i);
  return -1;
}

// An empty function produces no partitions regardless of maxQubits.
//
//   func.func @empty() { return }
TEST(GreedyOpPartitioner, EmptyFunctionNoPartitions) {
  MLIRContext ctx;
  auto mod = parse(ctx, R"(func.func @empty() { return })");
  ASSERT_TRUE(mod);
  cudaq::opt::GreedyOpPartitioner p(getFunc(*mod), 2);
  EXPECT_TRUE(p.getPartitions().empty());
}

// All gates on a single qubit fit in one partition.
//
//   %w = quake.null_wire
//   %h = quake.h %w
//   %x = quake.x %h
//   %t = quake.t %x
//
// Expected: 1 partition containing all four ops.
TEST(GreedyOpPartitioner, SingleChainOnePartition) {
  MLIRContext ctx;
  auto mod = parse(ctx, R"(
    func.func @k() {
      %w = quake.null_wire
      %h = quake.h %w : (!quake.wire) -> !quake.wire
      %x = quake.x %h : (!quake.wire) -> !quake.wire
      %t = quake.t %x : (!quake.wire) -> !quake.wire
      return
    }
  )");
  ASSERT_TRUE(mod);
  cudaq::opt::GreedyOpPartitioner p(getFunc(*mod), 2);
  EXPECT_EQ(p.getPartitions().size(), 1u);
  EXPECT_EQ(p.getPartitions()[0].size(), 4u);
}

// Two qubits within maxQubits stay in one partition even when interleaved.
//
//   %w0 = quake.null_wire
//   %w1 = quake.null_wire
//   %h0 = quake.h %w0
//   %h1 = quake.h %w1
//   %cx:2 = quake.x [%h0] %h1
//
// Expected (maxQubits=2): 1 partition with all five ops.
TEST(GreedyOpPartitioner, TwoQubitsWithinLimit) {
  MLIRContext ctx;
  auto mod = parse(ctx, R"(
    func.func @k() {
      %w0 = quake.null_wire
      %w1 = quake.null_wire
      %h0 = quake.h %w0 : (!quake.wire) -> !quake.wire
      %h1 = quake.h %w1 : (!quake.wire) -> !quake.wire
      %cx:2 = quake.x [%h0] %h1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      return
    }
  )");
  ASSERT_TRUE(mod);
  cudaq::opt::GreedyOpPartitioner p(getFunc(*mod), 2);
  ASSERT_EQ(p.getPartitions().size(), 1u);
  EXPECT_EQ(p.getPartitions()[0].size(), 5u);
}

// When a gate bridges a full partition's qubit to a new one, only the full
// partition is closed; the smaller partition survives and absorbs the bridging
// gate together with its own qubits.
//
//   %w0, %w1, %w2 = null_wire x3
//   null_wire(%w0), null_wire(%w1) → P1 (qubitCount=2, full at maxQubits=2)
//   null_wire(%w2)                 → P2 (qubitCount=1, P1 was full)
//   %cx:2  = quake.x [%w0] %w1    → touches P1 only; P1 not yet over limit →
//   add to P1 %cx2:2 = quake.x [%cx#0] %w2  → touches P1 (cx#0) and P2 (w2)
//                                    mergedCount=3 > 2; close P1 (largest),
//                                    extQubits+=1 survivors={P2}, P2.count+1=2
//                                    ≤ 2 → add cx2 to P2
//
// Expected: 2 partitions.
//   P1 = { null_wire(%w0), null_wire(%w1), cx }
//   P2 = { null_wire(%w2), cx2 }
TEST(GreedyOpPartitioner, FullGroupConnectingToNewQubitCloses) {
  MLIRContext ctx;
  auto mod = parse(ctx, R"(
    func.func @k() {
      %w0 = quake.null_wire
      %w1 = quake.null_wire
      %w2 = quake.null_wire
      %cx:2  = quake.x [%w0] %w1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      %cx2:2 = quake.x [%cx#0] %w2 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      return
    }
  )");
  ASSERT_TRUE(mod);

  Operation *cx = nullptr, *cx2 = nullptr;
  unsigned nullWires = 0;
  mod->walk([&](Operation *op) {
    if (isa<cudaq::quake::NullWireOp>(op))
      ++nullWires;
    else if (isa<cudaq::quake::XOp>(op)) {
      if (!cx)
        cx = op;
      else
        cx2 = op;
    }
  });
  ASSERT_EQ(nullWires, 3u);
  ASSERT_TRUE(cx && cx2);

  cudaq::opt::GreedyOpPartitioner p(getFunc(*mod), 2);
  ASSERT_EQ(p.getPartitions().size(), 2u);

  // cx fills P1; cx2 joins P2 (the smaller survivor after P1 closes).
  int cxPart = partitionOf(p.getPartitions(), cx);
  int cx2Part = partitionOf(p.getPartitions(), cx2);
  EXPECT_GE(cxPart, 0);
  EXPECT_GE(cx2Part, 0);
  EXPECT_NE(cxPart, cx2Part);
}

// KEY TEST: Gates on disjoint qubit chains interleaved in block order should
// not force each other's partitions to close. Each chain should stay in its
// own open partition independently.
//
//   %w0, %w1 = null_wire x2   → P1 = {w0, w1}  (full at maxQubits=2)
//   %w2      = null_wire       → P2 = {w2}
//   %h0 = quake.h %w0          → P1
//   %h2 = quake.h %w2          → P2  (must NOT close P1)
//   %h1 = quake.h %w1          → P1  (still open)
//
// Expected: 2 partitions, h0 and h1 in the same partition, h2 in a different
// one.
TEST(GreedyOpPartitioner, DisjointChainsInterleavedStaySeparate) {
  MLIRContext ctx;
  auto mod = parse(ctx, R"(
    func.func @k() {
      %w0 = quake.null_wire
      %w1 = quake.null_wire
      %w2 = quake.null_wire
      %h0 = quake.h %w0 : (!quake.wire) -> !quake.wire
      %h2 = quake.h %w2 : (!quake.wire) -> !quake.wire
      %h1 = quake.h %w1 : (!quake.wire) -> !quake.wire
      return
    }
  )");
  ASSERT_TRUE(mod);

  SmallVector<Operation *> hOps;
  mod->walk([&](cudaq::quake::HOp h) { hOps.push_back(h.getOperation()); });
  ASSERT_EQ(hOps.size(), 3u);
  // Block order: h0=hOps[0], h2=hOps[1], h1=hOps[2].
  Operation *h0 = hOps[0], *h2 = hOps[1], *h1 = hOps[2];

  cudaq::opt::GreedyOpPartitioner p(getFunc(*mod), 2);
  ASSERT_EQ(p.getPartitions().size(), 2u);

  int p0 = partitionOf(p.getPartitions(), h0);
  int p1 = partitionOf(p.getPartitions(), h1);
  int p2 = partitionOf(p.getPartitions(), h2);
  EXPECT_GE(p0, 0);
  EXPECT_GE(p1, 0);
  EXPECT_GE(p2, 0);
  // h0 and h1 (both on the w0/w1 chain) must be in the same partition.
  EXPECT_EQ(p0, p1);
  // h2 (on the w2 chain) must be in a different partition.
  EXPECT_NE(p0, p2);
}

// Two partitions that both touch a bridging gate can be merged when their
// combined qubit count fits within maxQubits.
//
// With maxQubits=1, each qubit starts its own partition. A CX gate bridges
// two size-1 partitions whose merge (1+1=2) exceeds maxQubits=1, so both
// close and a fresh partition starts.
//
// With maxQubits=2, the two null_wires land in the SAME initial partition, so
// the CX is single-partition. Use maxQubits=1 to force two separate
// size-1 partitions, then bridge them.
//
//   maxQubits=1:
//   %w0 = null_wire → P1={w0}
//   %w1 = null_wire → P2={w1}   (P1 full at 1)
//   %cx:2 = quake.x [%w0] %w1  → mergedCount=1+1=2 > 1 → close P1,P2;
//   P3={cx,count=2}
//
// Expected: 3 partitions.
TEST(GreedyOpPartitioner, PartitionsMergedOrClosedAcrossBridge) {
  MLIRContext ctx;
  auto mod = parse(ctx, R"(
    func.func @k() {
      %w0 = quake.null_wire
      %w1 = quake.null_wire
      %cx:2 = quake.x [%w0] %w1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      return
    }
  )");
  ASSERT_TRUE(mod);

  Operation *cx = nullptr;
  mod->walk([&](cudaq::quake::XOp x) { cx = x.getOperation(); });
  ASSERT_TRUE(cx);

  // maxQubits=1: both null_wires in separate partitions; cx exceeds limit → 3
  // parts.
  cudaq::opt::GreedyOpPartitioner p1(getFunc(*mod), 1);
  EXPECT_EQ(p1.getPartitions().size(), 3u);

  // maxQubits=2: both null_wires and cx fit in one partition → 1 part.
  cudaq::opt::GreedyOpPartitioner p2(getFunc(*mod), 2);
  EXPECT_EQ(p2.getPartitions().size(), 1u);
}

// Every wire-bearing op is assigned to exactly one partition.
//
// Verifies that no op appears in more than one partition and that the union of
// all partition ops equals the set of wire-bearing ops in the function.
TEST(GreedyOpPartitioner, AllWireOpsAccountedFor) {
  MLIRContext ctx;
  auto mod = parse(ctx, R"(
    func.func @k() {
      %w0 = quake.null_wire
      %w1 = quake.null_wire
      %w2 = quake.null_wire
      %h0 = quake.h %w0 : (!quake.wire) -> !quake.wire
      %h2 = quake.h %w2 : (!quake.wire) -> !quake.wire
      %h1 = quake.h %w1 : (!quake.wire) -> !quake.wire
      %cx:2 = quake.x [%h0] %h1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      return
    }
  )");
  ASSERT_TRUE(mod);

  SmallVector<Operation *> wireOps;
  auto wireTy = cudaq::quake::WireType::get(&ctx);
  mod->walk([&](Operation *op) {
    bool hasWire =
        llvm::any_of(op->getOperands(),
                     [&](Value v) { return v.getType() == wireTy; }) ||
        llvm::any_of(op->getResults(),
                     [&](Value v) { return v.getType() == wireTy; });
    if (hasWire)
      wireOps.push_back(op);
  });

  cudaq::opt::GreedyOpPartitioner p(getFunc(*mod), 2);

  // Every wire op appears in exactly one partition.
  for (Operation *op : wireOps)
    EXPECT_GE(partitionOf(p.getPartitions(), op), 0)
        << "op not assigned to any partition";

  // No op appears twice.
  DenseSet<Operation *> seen;
  for (const auto &part : p.getPartitions())
    for (Operation *op : part)
      EXPECT_TRUE(seen.insert(op).second)
          << "op appears in multiple partitions";
}
