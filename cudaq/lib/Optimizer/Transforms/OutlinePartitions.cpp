/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Analysis/GreedyOpPartitioner.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_OUTLINEPARTITIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace cudaq::opt {
FailureOr<cc::CreateLambdaOp> outlinePartition(const DenseSet<Operation *> &);
LogicalResult outlinePartitions(ArrayRef<DenseSet<Operation *>>);
} // namespace cudaq::opt

static inline bool isWire(Value v) {
  return isa<cudaq::quake::WireType>(v.getType());
}

#ifndef NDEBUG
static bool validatePartition(const DenseSet<Operation *> &partition,
                              const SmallVector<Operation *> &orderedOps,
                              Block *block, const SetVector<Value> &outputs) {
  if (partition.empty())
    return false;
  for (Operation *op : partition) {
    if (op->getBlock() != block)
      return false;
    for (Value v : llvm::concat<Value>(op->getOperands(), op->getResults()))
      if (isa<cudaq::quake::RefType>(v.getType()))
        return false;
  }
  // Contiguity: no non-partition op lies on a wire path between two partition
  // ops.
  DenseSet<Value> seen;
  SmallVector<Value> worklist;
  for (Operation *op : orderedOps)
    for (Value res : op->getResults())
      if (isWire(res))
        for (Operation *user : res.getUsers())
          if (!partition.contains(user))
            for (Value r : user->getResults())
              if (isWire(r) && seen.insert(r).second)
                worklist.push_back(r);
  while (!worklist.empty()) {
    Value wire = worklist.pop_back_val();
    for (Operation *user : wire.getUsers()) {
      if (partition.contains(user))
        return false;
      for (Value res : user->getResults())
        if (isWire(res) && seen.insert(res).second)
          worklist.push_back(res);
    }
  }
  // All external wire consumers must live in the same block.
  for (Value out : outputs)
    for (Operation *user : out.getUsers())
      if (!partition.contains(user) && user->getBlock() != block)
        return false;
  return true;
}
#endif

FailureOr<cudaq::cc::CreateLambdaOp>
cudaq::opt::outlinePartition(const DenseSet<Operation *> &partition) {
  Block *block = (*partition.begin())->getBlock();
  Operation *anchor = *partition.begin();
  for (Operation *op : partition)
    if (anchor->isBeforeInBlock(op))
      anchor = op;

  // Walk the partition in block (topological) order so cloned operands are
  // always already mapped.
  SmallVector<Operation *> orderedOps;
  for (Operation &op : *block)
    if (partition.contains(&op))
      orderedOps.push_back(&op);

  // Infer the wire boundary. Because wires are use-once, an output wire has at
  // most one (external) use, so inputs and outputs cannot alias.
  SetVector<Value> inputs, outputs;
  for (Operation *op : orderedOps) {
    for (Value operand : op->getOperands()) {
      if (!isWire(operand))
        continue;
      Operation *def = operand.getDefiningOp();
      if (!def || !partition.contains(def))
        inputs.insert(operand);
    }
    for (Value res : op->getResults()) {
      if (!isWire(res))
        continue;
      bool external = res.use_empty();
      for (Operation *user : res.getUsers())
        if (!partition.contains(user)) {
          external = true;
          break;
        }
      if (external)
        outputs.insert(res);
    }
  }

  assert(validatePartition(partition, orderedOps, block, outputs) &&
         "invalid partition");

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

  OpBuilder builder(ctx);
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
  auto call = cudaq::cc::CallCallableOp::create(builder, loc, TypeRange(outTys),
                                                lambda.getResult(),
                                                ValueRange(inputList));

  for (auto [i, out] : llvm::enumerate(outputList))
    out.replaceAllUsesWith(call.getResult(i));
  for (Operation *op : llvm::reverse(orderedOps))
    op->erase();

  mlir::sortTopologically(block);
  return lambda;
}

LogicalResult
cudaq::opt::outlinePartitions(ArrayRef<DenseSet<Operation *>> partitions) {
  auto result = success();
  for (const auto &partition : partitions)
    if (failed(outlinePartition(partition)))
      result = failure();
  return result;
}

namespace {
class OutlinePartitionsPass
    : public cudaq::opt::impl::OutlinePartitionsBase<OutlinePartitionsPass> {
  using Base = cudaq::opt::impl::OutlinePartitionsBase<OutlinePartitionsPass>;

public:
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    if (strategy == "greedy") {
      cudaq::opt::GreedyOpPartitioner analysis(op, maxQubits);
      if (failed(cudaq::opt::outlinePartitions(analysis.getPartitions())))
        signalPassFailure();
      return;
    }
    op->emitError("outline-partitions: unknown strategy '") << strategy << "'";
    signalPassFailure();
  }
};
} // namespace

namespace cudaq::opt {
std::unique_ptr<mlir::Pass> createOutlinePartitionsPass() {
  return std::make_unique<OutlinePartitionsPass>();
}
} // namespace cudaq::opt
