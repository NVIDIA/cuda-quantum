/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_OUTLINEPARTITIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
static bool isWire(Value v) { return isa<cudaq::quake::WireType>(v.getType()); }
} // namespace

FailureOr<cudaq::cc::CreateLambdaOp>
cudaq::opt::outlinePartition(ArrayRef<Operation *> partition) {
  if (partition.empty())
    return failure();

  DenseSet<Operation *> inPartition(partition.begin(), partition.end());

  // All operations must live in one block; find the last one in block order to
  // use as the anchor for the closure and its application.
  Block *block = partition.front()->getBlock();
  Operation *anchor = partition.front();
  for (Operation *op : partition) {
    if (op->getBlock() != block)
      return failure();
    if (anchor->isBeforeInBlock(op))
      anchor = op;
  }

  // Walk the partition in block (topological) order so cloned operands are
  // always already mapped.
  SmallVector<Operation *> orderedOps;
  for (Operation &op : *block)
    if (inPartition.contains(&op))
      orderedOps.push_back(&op);

  // Infer the wire boundary. Because wires are use-once, an output wire has at
  // most one (external) use, so inputs and outputs cannot alias.
  SetVector<Value> inputs, outputs;
  for (Operation *op : orderedOps) {
    for (Value operand : op->getOperands()) {
      if (!isWire(operand))
        continue;
      Operation *def = operand.getDefiningOp();
      if (!def || !inPartition.contains(def))
        inputs.insert(operand);
    }
    for (Value res : op->getResults()) {
      if (!isWire(res))
        continue;
      bool external = res.use_empty();
      for (Operation *user : res.getUsers())
        if (!inPartition.contains(user)) {
          external = true;
          break;
        }
      if (external)
        outputs.insert(res);
    }
  }

  // Contiguity check: following each wire that exits the partition, ensure no
  // descendant wire re-enters it (i.e., no non-partition op lies on a wire
  // path between two partition ops).
  {
    DenseSet<Value> seen;
    SmallVector<Value> worklist;
    for (Operation *op : orderedOps)
      for (Value res : op->getResults())
        if (isWire(res))
          for (Operation *user : res.getUsers())
            if (!inPartition.contains(user))
              for (Value r : user->getResults())
                if (isWire(r) && seen.insert(r).second)
                  worklist.push_back(r);
    while (!worklist.empty()) {
      Value wire = worklist.pop_back_val();
      for (Operation *user : wire.getUsers()) {
        if (inPartition.contains(user))
          return failure();
        for (Value res : user->getResults())
          if (isWire(res) && seen.insert(res).second)
            worklist.push_back(res);
      }
    }
  }

  // Require all external wire consumers to live in the same block.
  for (Value out : outputs)
    for (Operation *user : out.getUsers())
      if (!inPartition.contains(user) && user->getBlock() != block)
        return failure();

  auto inputList = inputs.takeVector();
  auto outputList = outputs.takeVector();

  auto *ctx = block->getParentOp()->getContext();
  SmallVector<Type> inTys, outTys;
  for (Value v : inputList)
    inTys.push_back(v.getType());
  for (Value v : outputList)
    outTys.push_back(v.getType());
  auto callableTy =
      cudaq::cc::CallableType::get(ctx, FunctionType::get(ctx, inTys, outTys));

  // Place lambda/call before the earliest external wire consumer so that the
  // call results dominate their uses after the partition ops are erased.
  OpBuilder builder(ctx);
  Operation *insertBefore = nullptr;
  for (Value out : outputList)
    for (Operation *user : out.getUsers())
      if (!inPartition.contains(user))
        if (!insertBefore || user->isBeforeInBlock(insertBefore))
          insertBefore = user;
  if (insertBefore)
    builder.setInsertionPoint(insertBefore);
  else
    builder.setInsertionPointAfter(anchor);
  Location loc = anchor->getLoc();

  auto lambda = cudaq::cc::CreateLambdaOp::create(
      builder, loc, callableTy, [&](OpBuilder &b, Location l) {
        Block *body = b.getInsertionBlock();
        IRMapping map;
        for (auto [i, in] : llvm::enumerate(inputList))
          map.map(in, body->getArgument(i));
        for (Operation *op : orderedOps)
          b.clone(*op, map);
        SmallVector<Value> results;
        for (Value out : outputList)
          results.push_back(map.lookup(out));
        cudaq::cc::ReturnOp::create(b, l, results);
      });

  builder.setInsertionPointAfter(lambda);
  auto call = cudaq::cc::CallCallableOp::create(
      builder, loc, TypeRange(outTys), lambda.getResult(),
      ValueRange(inputList));

  for (auto [i, out] : llvm::enumerate(outputList))
    out.replaceAllUsesWith(call.getResult(i));
  for (Operation *op : llvm::reverse(orderedOps))
    op->erase();

  return lambda;
}

LogicalResult cudaq::opt::outlinePartitions(
    ArrayRef<SmallVector<Operation *>> partitions) {
  auto result = success();
  for (const auto &partition : partitions)
    if (failed(outlinePartition(partition)))
      result = failure();
  return result;
}

struct OutlinePartitionsAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlinePartitionsAnalysis)

  explicit OutlinePartitionsAnalysis(Operation *op) {
    // TODO: implement partitioning logic
  }

  ArrayRef<SmallVector<Operation *>> getPartitions() const { return partitions; }

private:
  SmallVector<SmallVector<Operation *>> partitions;
};

namespace {
class OutlinePartitionsPass
    : public cudaq::opt::impl::OutlinePartitionsBase<OutlinePartitionsPass> {
  using Base = cudaq::opt::impl::OutlinePartitionsBase<OutlinePartitionsPass>;

public:
  using Base::Base;

  void runOnOperation() override {
    auto &analysis = getAnalysis<OutlinePartitionsAnalysis>();
    if (failed(cudaq::opt::outlinePartitions(analysis.getPartitions())))
      signalPassFailure();
  }
};
} // namespace

namespace cudaq::opt {
std::unique_ptr<mlir::Pass> createOutlinePartitionsPass() {
  return std::make_unique<OutlinePartitionsPass>();
}
} // namespace cudaq::opt
