/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "OutlinePartitions.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

// Accept wire operations and scalar discrimination with no memory
// effects.
static bool isSupportedOutlineOp(Operation *op) {
  return isMemoryEffectFree(op) &&
         (isa<cudaq::quake::DiscriminateOp>(op) ||
          cudaq::quake::detail::getWireFlow(op).has_value());
}

static FailureOr<cudaq::cc::CreateLambdaOp>
outlinePartitionImpl(ArrayRef<Operation *> partition,
                     const DominanceInfo &dominance) {
  if (partition.empty())
    return failure();

  DenseSet<Operation *> inPartition(partition.begin(), partition.end());

  // All operations must be supported and live in one block.
  Block *block = partition.front()->getBlock();
  for (Operation *op : partition) {
    if (op->getBlock() != block)
      return failure();
    if (!isSupportedOutlineOp(op))
      return failure();
  }

  // Sort the unique selected operations in block (topological) order so
  // selected definitions are cloned before their users.
  SmallVector<Operation *> orderedOps(inPartition.begin(), inPartition.end());
  llvm::sort(orderedOps, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  Operation *anchor = orderedOps.back();

  // Collect captured inputs and outputs. Non-wire outputs precede wires in the
  // callable results.
  SetVector<Value> inputs, outputs, nonWireInputs, nonWireOutputs;
  for (Operation *op : orderedOps) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !inPartition.contains(def)) {
        if (isa<cudaq::quake::WireType>(operand.getType()))
          inputs.insert(operand);
        else
          nonWireInputs.insert(operand);
      }
    }
    for (Value res : op->getResults()) {
      // Preserve unused wires as outputs.
      bool externalUse =
          isa<cudaq::quake::WireType>(res.getType()) && res.use_empty();
      // Export any result with a user outside the partition.
      for (Operation *user : res.getUsers())
        if (!inPartition.contains(user)) {
          externalUse = true;
          break;
        }
      if (externalUse) {
        if (isa<cudaq::quake::WireType>(res.getType()))
          outputs.insert(res);
        else
          nonWireOutputs.insert(res);
      }
    }
  }

  // Require all external output consumers to live in the same block.
  for (Value out : llvm::concat<const Value>(nonWireOutputs, outputs))
    for (Operation *user : out.getUsers())
      if (!inPartition.contains(user) && user->getBlock() != block)
        return failure();

  auto inputList = inputs.takeVector();
  auto nonWireInputList = nonWireInputs.takeVector();
  auto nonWireOutputList = nonWireOutputs.takeVector();

  // Create a map to order the output wires in the same order as the
  // discovered wire inputs by tracing wire flow
  // current wire --> original input index
  DenseMap<Value, unsigned> outputIdxByWire;
  // create vector of ordered wire outputs to match input wire order
  SmallVector<Value> orderedWireOutputs(inputList.begin(), inputList.end());

  for (auto [i, wire] : llvm::enumerate(inputList))
    outputIdxByWire[wire] = i;

  // now order the outputs based on discovered wire inputs
  for (Operation *op : orderedOps) {
    // no wire input in `quake.discriminate`
    if (isa<cudaq::quake::DiscriminateOp>(op))
      continue;

    auto flow = cudaq::quake::detail::getWireFlow(op);
    if (!flow)
      return failure();

    for (auto [in, out] : llvm::zip(flow->inputs, flow->results)) {
      auto it = outputIdxByWire.find(in);
      if (it == outputIdxByWire.end())
        return failure();

      unsigned idx = it->second;
      outputIdxByWire.erase(it);
      outputIdxByWire[out] = idx;
      orderedWireOutputs[idx] = out;
    }
  }

  // Check that tracing found exactly the discovered output wires.
  if (orderedWireOutputs.size() != outputs.size())
    return failure();

  for (Value out : orderedWireOutputs)
    if (!out || !outputs.contains(out))
      return failure();

  // Prepare a callable without explicit parameters. Non-wire results precede
  // wire results.
  auto *ctx = block->getParentOp()->getContext();
  SmallVector<Type> outTys;
  for (Value v : nonWireOutputList)
    outTys.push_back(v.getType());
  for (Value v : orderedWireOutputs)
    outTys.push_back(v.getType());
  auto callableTy =
      cudaq::cc::CallableType::get(ctx, FunctionType::get(ctx, {}, outTys));

  // Insert before the earliest external consumer, or after the last selected
  // operation if there are no consumers. Reject placements where an input is
  // unavailable.
  OpBuilder builder(ctx);
  Operation *insertBefore = nullptr;
  for (Value out : llvm::concat<Value>(nonWireOutputList, orderedWireOutputs))
    for (Operation *user : out.getUsers())
      if (!inPartition.contains(user))
        if (!insertBefore || user->isBeforeInBlock(insertBefore))
          insertBefore = user;
  for (Value in : llvm::concat<Value>(nonWireInputList, inputList)) {
    bool available = insertBefore
                         ? dominance.properlyDominates(in, insertBefore)
                         : dominance.dominates(in, anchor);
    if (!available)
      return failure();
  }

  if (insertBefore)
    builder.setInsertionPoint(insertBefore);
  else
    builder.setInsertionPointAfter(anchor);
  Location loc = anchor->getLoc();

  // Create the lambda with all external inputs captured.
  auto lambda = cudaq::cc::CreateLambdaOp::create(
      builder, loc, callableTy, [&](OpBuilder &b, Location l) {
        IRMapping map;
        // External inputs remain captures; cloning maps internally defined
        // values to their copies.
        for (Operation *op : orderedOps)
          b.clone(*op, map);
        SmallVector<Value> results;
        for (Value out : nonWireOutputList)
          results.push_back(map.lookup(out));
        for (Value out : orderedWireOutputs)
          results.push_back(map.lookup(out));
        cudaq::cc::ReturnOp::create(b, l, results);
      });

  // Call the lambda without explicit arguments.
  builder.setInsertionPointAfter(lambda);
  OperationState state(loc, cudaq::cc::CallCallableOp::getOperationName());
  state.addTypes(outTys);
  state.addOperands(lambda.getResult());
  auto call = cast<cudaq::cc::CallCallableOp>(builder.create(state));

  for (auto [i, out] : llvm::enumerate(nonWireOutputList))
    out.replaceAllUsesWith(call.getResult(i));
  for (auto [i, out] : llvm::enumerate(orderedWireOutputs))
    out.replaceAllUsesWith(call.getResult(i + nonWireOutputList.size()));
  for (Operation *op : llvm::reverse(orderedOps))
    op->erase();

  return lambda;
}

FailureOr<cudaq::cc::CreateLambdaOp>
cudaq::opt::outlinePartition(ArrayRef<Operation *> partition) {
  DominanceInfo dominance;
  return outlinePartitionImpl(partition, dominance);
}

LogicalResult
cudaq::opt::outlinePartitions(ArrayRef<SmallVector<Operation *>> partitions) {
  // The CFG of existing regions is unchanged, so cached dominance information
  // remains valid across these rewrites.
  DominanceInfo dominance;
  auto result = success();
  for (const auto &partition : partitions)
    if (failed(outlinePartitionImpl(partition, dominance)))
      result = failure();
  return result;
}
