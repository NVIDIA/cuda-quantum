/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/PatternMatch.h"
#include <cassert>
#include <cstddef>

namespace cudaq::opt::decomp {
using namespace mlir;

/// @brief This is a wrapper class for `PatternRewriter::create<>()` for
/// `QuakeOperator`s. If the controls and targets are `cudaq::quake::WireType`,
/// then this wrapper class's methods update the controls and targets in the
/// `create` calls to the corresponding wires in the output. If they are NOT
/// `WireType`, then the creates behave the exact same as a regular
/// `PatternRewriter`.
class QuakeOperatorCreator {
public:
  QuakeOperatorCreator(PatternRewriter &rewriter) : rewriter(rewriter) {}

  /// Construct a resultType (suitable to be pass into the `TypeRange wires`
  /// builder for cases when you have one input ValueRange.
  SmallVector<Type> getResultType(ValueRange operands) {
    std::size_t numOutputWires = llvm::count_if(operands, [](const Value &v) {
      return isa<cudaq::quake::WireType>(v.getType());
    });

    return SmallVector<Type>(
        numOutputWires, cudaq::quake::WireType::get(rewriter.getContext()));
  }

  /// Construct a resultType (suitable to be pass into the `TypeRange wires`
  /// builder for cases when you have two input ValueRanges.
  SmallVector<Type> getResultType(ValueRange operands1, ValueRange operands2) {
    std::size_t numOutputWires =
        llvm::count_if(operands1,
                       [](const Value &v) {
                         return isa<cudaq::quake::WireType>(v.getType());
                       }) +
        llvm::count_if(operands2, [](const Value &v) {
          return isa<cudaq::quake::WireType>(v.getType());
        });

    return SmallVector<Type>(
        numOutputWires, cudaq::quake::WireType::get(rewriter.getContext()));
  }

  /// Pluck out the values from \p newValues whose type is `WireType` and
  /// replace all the \p op uses with those values.
  void selectWiresAndReplaceUses(Operation *op, ValueRange newValues) {
    SmallVector<Value, 4> newWireValues;
    for (const auto &v : newValues)
      if (isa<cudaq::quake::WireType>(v.getType()))
        newWireValues.push_back(v);
    assert(op->getResults().size() == newWireValues.size() &&
           "incorrect number of output wires provided");
    op->replaceAllUsesWith(newWireValues);
  }

  /// Pluck out the values from \p controls and \p target whose type is
  /// `WireType` and replace all the \p op uses with those values.
  void selectWiresAndReplaceUses(Operation *op, ValueRange controls,
                                 Value target) {
    SmallVector<Value, 4> newWireValues;
    for (const auto &v : controls)
      if (isa<cudaq::quake::WireType>(v.getType()))
        newWireValues.push_back(v);
    if (isa<cudaq::quake::WireType>(target.getType()))
      newWireValues.push_back(target);
    assert(op->getResults().size() == newWireValues.size() &&
           "incorrect number of output wires provided");
    op->replaceAllUsesWith(newWireValues);
  }

  template <typename OpTy>
  OpTy create(Location location, Value &target) {
    OpTy op;
    op = OpTy::create(rewriter, location, getResultType(target), false,
                      ValueRange{}, ValueRange{}, target, DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    if (isa<cudaq::quake::WireType>(target.getType()) &&
        resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, bool is_adj, Value &target) {
    OpTy op;
    op = OpTy::create(rewriter, location, getResultType(target), is_adj,
                      ValueRange{}, ValueRange{}, target, DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    if (isa<cudaq::quake::WireType>(target.getType()) &&
        resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, Value &control, Value &target) {
    OpTy op;
    op = OpTy::create(rewriter, location, getResultType(control, target), false,
                      ValueRange{}, control, target, DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    if (isa<cudaq::quake::WireType>(control.getType()) &&
        resultIt != resultWiresEnd)
      control = *resultIt++;
    if (isa<cudaq::quake::WireType>(target.getType()) &&
        resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, bool is_adj, ValueRange parameters,
              SmallVectorImpl<Value> &controls, Value &target) {
    OpTy op;
    op = OpTy::create(rewriter, location, getResultType(controls, target),
                      is_adj, parameters, controls, target,
                      DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    for (auto &c : controls)
      if (isa<cudaq::quake::WireType>(c.getType()) &&
          resultIt != resultWiresEnd)
        c = *resultIt++;
    if (isa<cudaq::quake::WireType>(target.getType()) &&
        resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, ValueRange parameters,
              SmallVectorImpl<Value> &controls, Value &target) {
    OpTy op;
    op =
        OpTy::create(rewriter, location, getResultType(controls, target), false,
                     parameters, controls, target, DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    for (auto &c : controls)
      if (isa<cudaq::quake::WireType>(c.getType()) &&
          resultIt != resultWiresEnd)
        c = *resultIt++;
    if (isa<cudaq::quake::WireType>(target.getType()) &&
        resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, SmallVectorImpl<Value> &controls,
              Value &target) {
    OpTy op;
    op =
        OpTy::create(rewriter, location, getResultType(controls, target), false,
                     ValueRange{}, controls, target, DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    for (auto &c : controls)
      if (isa<cudaq::quake::WireType>(c.getType()) &&
          resultIt != resultWiresEnd)
        c = *resultIt++;
    if (isa<cudaq::quake::WireType>(target.getType()) &&
        resultIt != resultWiresEnd)
      target = *resultIt;
    return op;
  }

  template <typename OpTy>
  OpTy create(Location location, SmallVectorImpl<Value> &targets) {
    OpTy op;
    op =
        OpTy::create(rewriter, location, getResultType(targets), false,
                     ValueRange{}, ValueRange{}, targets, DenseBoolArrayAttr{});
    auto resultWires = op.getWires();
    auto resultIt = resultWires.begin();
    auto resultWiresEnd = resultWires.end();
    for (auto &t : targets)
      if (isa<cudaq::quake::WireType>(t.getType()) &&
          resultIt != resultWiresEnd)
        t = *resultIt++;
    return op;
  }

private:
  PatternRewriter &rewriter;
};

} // namespace cudaq::opt::decomp
