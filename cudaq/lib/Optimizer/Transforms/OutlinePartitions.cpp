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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace {
static bool isWire(Value v) { return isa<cudaq::quake::WireType>(v.getType()); }
static auto filterWireValues(ValueRange values) {
  return llvm::make_filter_range(values, isWire);
}

// Accept scalar-wire operations and scalar discrimination with no memory
// effects.
static bool isSupportedOutlineOp(Operation *op) {
  return isMemoryEffectFree(op) &&
         (isa<cudaq::quake::DiscriminateOp>(op) ||
          cudaq::quake::detail::getScalarWireFlow(op).has_value());
}
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
    if (!isSupportedOutlineOp(op))
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

  // Collect the input/output boundary, keeping non-wire values separate so
  // they can precede wires in the callable signature.
  SetVector<Value> inputs, outputs, nonWireInputs, nonWireOutputs;
  for (Operation *op : orderedOps) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !inPartition.contains(def)) {
        if (isWire(operand))
          inputs.insert(operand);
        else
          nonWireInputs.insert(operand);
      }
    }
    for (Value res : op->getResults()) {
      // Preserve unused wires as outputs.
      bool externalUse = isWire(res) && res.use_empty();
      // Export any result with a user outside the partition.
      for (Operation *user : res.getUsers())
        if (!inPartition.contains(user)) {
          externalUse = true;
          break;
        }
      if (externalUse) {
        if (isWire(res))
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
  auto outputList = outputs.takeVector();
  auto nonWireInputList = nonWireInputs.takeVector();
  auto nonWireOutputList = nonWireOutputs.takeVector();

  // Prepare the callable type with non-wire values first.
  auto *ctx = block->getParentOp()->getContext();
  SmallVector<Type> inTys, outTys;
  for (Value v : nonWireInputList)
    inTys.push_back(v.getType());
  for (Value v : inputList)
    inTys.push_back(v.getType());
  for (Value v : nonWireOutputList)
    outTys.push_back(v.getType());
  for (Value v : outputList)
    outTys.push_back(v.getType());
  auto callableTy =
      cudaq::cc::CallableType::get(ctx, FunctionType::get(ctx, inTys, outTys));

  // Insert before the earliest external consumer, or after the last selected
  // operation if there are no consumers. Reject placements where an input is
  // unavailable.
  OpBuilder builder(ctx);
  Operation *insertBefore = nullptr;
  for (Value out : llvm::concat<Value>(nonWireOutputList, outputList))
    for (Operation *user : out.getUsers())
      if (!inPartition.contains(user))
        if (!insertBefore || user->isBeforeInBlock(insertBefore))
          insertBefore = user;
  DominanceInfo dominance;
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

  // Create the lambda.
  auto lambda = cudaq::cc::CreateLambdaOp::create(
      builder, loc, callableTy, [&](OpBuilder &b, Location l) {
        Block *body = b.getInsertionBlock();
        IRMapping map;
        for (auto [i, in] : llvm::enumerate(nonWireInputList))
          map.map(in, body->getArgument(i));
        for (auto [i, in] : llvm::enumerate(inputList))
          map.map(in, body->getArgument(i + nonWireInputList.size()));
        for (Operation *op : orderedOps)
          b.clone(*op, map);
        SmallVector<Value> results;
        for (Value out : nonWireOutputList)
          results.push_back(map.lookup(out));
        for (Value out : outputList)
          results.push_back(map.lookup(out));
        cudaq::cc::ReturnOp::create(b, l, results);
      });

  // Call the lambda with non-wire arguments followed by wires.
  builder.setInsertionPointAfter(lambda);
  OperationState state(loc, cudaq::cc::CallCallableOp::getOperationName());
  state.addTypes(outTys);
  state.addOperands(lambda.getResult());
  state.addOperands(nonWireInputList);
  state.addOperands(inputList);
  auto call = cast<cudaq::cc::CallCallableOp>(builder.create(state));

  for (auto [i, out] : llvm::enumerate(nonWireOutputList))
    out.replaceAllUsesWith(call.getResult(i));
  for (auto [i, out] : llvm::enumerate(outputList))
    out.replaceAllUsesWith(call.getResult(i + nonWireOutputList.size()));
  for (Operation *op : llvm::reverse(orderedOps))
    op->erase();

  return lambda;
}

LogicalResult
cudaq::opt::outlinePartitions(ArrayRef<SmallVector<Operation *>> partitions) {
  auto result = success();
  for (const auto &partition : partitions)
    if (failed(outlinePartition(partition)))
      result = failure();
  return result;
}
