//===- ExpandQuakeRegisterTraversals.cpp - Quake traversal prep -*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#include "qlx/Conversion/QuakeToQLXPasses.h"

#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

namespace qlx {
#define GEN_PASS_DEF_EXPANDQUAKEREGISTERTRAVERSALS
#include "qlx/Conversion/QuakeToQLXPasses.h.inc"
} // namespace qlx

using namespace mlir;

namespace {

static std::optional<int64_t> constInt(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  return std::nullopt;
}

struct FastCountedLoop {
  uint64_t tripCount;
};

// Match the narrow normalized form emitted by cc-loop-normalize. Keeping this
// proof local and strict makes direct SSA cloning safe: the while region has no
// observable work to preserve, and every carried value is explicitly threaded
// through body and step terminators.
static FailureOr<FastCountedLoop> matchFastCountedLoop(cudaq::cc::LoopOp loop) {
  if (loop.isPostConditional() || loop.hasPythonElse() || !loop.hasStep() ||
      loop.hasBreakInBody())
    return failure();
  if (!llvm::hasSingleElement(loop.getWhileRegion()) ||
      !llvm::hasSingleElement(loop.getBodyRegion()) ||
      !llvm::hasSingleElement(loop.getStepRegion()))
    return failure();

  Block *whileBlock = loop.getWhileBlock();
  Block *bodyBlock = loop.getDoEntryBlock();
  Block *stepBlock = loop.getStepBlock();
  auto condition =
      dyn_cast<cudaq::cc::ConditionOp>(whileBlock->getTerminator());
  auto bodyContinue =
      dyn_cast<cudaq::cc::ContinueOp>(bodyBlock->getTerminator());
  auto stepContinue =
      dyn_cast<cudaq::cc::ContinueOp>(stepBlock->getTerminator());
  if (!condition || !bodyContinue || !stepContinue)
    return failure();

  ValueRange initial = loop.getInitialArgs();
  auto whileArgs = whileBlock->getArguments();
  auto bodyArgs = bodyBlock->getArguments();
  auto stepArgs = stepBlock->getArguments();
  if (initial.size() != whileArgs.size() || initial.size() != bodyArgs.size() ||
      initial.size() != stepArgs.size() ||
      initial.size() != loop.getNumResults() ||
      condition.getResults().size() != initial.size() ||
      bodyContinue.getNumOperands() != initial.size() ||
      stepContinue.getNumOperands() != initial.size())
    return failure();
  for (auto [forwarded, argument] :
       llvm::zip_equal(condition.getResults(), whileArgs))
    if (forwarded != argument)
      return failure();

  // Only the comparison may precede cc.condition. Dropping the while region is
  // semantics-preserving after proving the constant trip count.
  if (std::distance(whileBlock->begin(), whileBlock->end()) != 2)
    return failure();
  auto compare = condition.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare || compare->getBlock() != whileBlock)
    return failure();
  auto predicate = compare.getPredicate();
  if (predicate != arith::CmpIPredicate::slt &&
      predicate != arith::CmpIPredicate::ult)
    return failure();

  unsigned induction = initial.size();
  for (unsigned index = 0; index < whileArgs.size(); ++index)
    if (compare.getLhs() == whileArgs[index]) {
      induction = index;
      break;
    }
  if (induction == initial.size())
    return failure();
  if (bodyContinue.getOperand(induction) != bodyArgs[induction])
    return failure();

  auto initialValue = constInt(initial[induction]);
  auto upperBound = constInt(compare.getRhs());
  if (!initialValue || !upperBound || *initialValue < 0 || *upperBound < 0)
    return failure();

  Value nextInduction = stepContinue.getOperand(induction);
  auto add = nextInduction.getDefiningOp<arith::AddIOp>();
  if (!add || add->getBlock() != stepBlock)
    return failure();
  Value stepValue;
  if (add.getLhs() == stepArgs[induction])
    stepValue = add.getRhs();
  else if (add.getRhs() == stepArgs[induction])
    stepValue = add.getLhs();
  else
    return failure();
  auto step = constInt(stepValue);
  if (!step || *step <= 0)
    return failure();

  uint64_t start = static_cast<uint64_t>(*initialValue);
  uint64_t stop = static_cast<uint64_t>(*upperBound);
  uint64_t stride = static_cast<uint64_t>(*step);
  uint64_t tripCount = start >= stop ? 0 : 1 + (stop - start - 1) / stride;

  // arith.addi is modular. Prove that the step after the final body cannot
  // wrap and make the source loop continue after the mathematical induction
  // has crossed its upper bound.
  auto inductionType = dyn_cast<IntegerType>(initial[induction].getType());
  if (!inductionType || inductionType.getWidth() > 64)
    return failure();
  constexpr unsigned proofWidth = 128;
  llvm::APInt finalInduction(proofWidth, start);
  finalInduction +=
      llvm::APInt(proofWidth, tripCount) * llvm::APInt(proofWidth, stride);
  llvm::APInt maximum =
      predicate == arith::CmpIPredicate::slt
          ? llvm::APInt::getSignedMaxValue(inductionType.getWidth())
                .zext(proofWidth)
          : llvm::APInt::getMaxValue(inductionType.getWidth()).zext(proofWidth);
  if (finalInduction.ugt(maximum))
    return failure();
  return FastCountedLoop{tripCount};
}

static bool blocksWireConversion(cudaq::cc::LoopOp loop) {
  bool found = false;
  loop->walk<WalkOrder::PreOrder>([&](Operation *operation) -> WalkResult {
    if (operation != loop.getOperation() && isa<cudaq::cc::LoopOp>(operation))
      return WalkResult::skip();
    if (auto extract = dyn_cast<cudaq::quake::ExtractRefOp>(operation);
        extract &&
        extract.getRawIndex() == cudaq::quake::ExtractRefOp::kDynamicIndex) {
      found = true;
      return WalkResult::interrupt();
    }
    if (auto slice = dyn_cast<cudaq::quake::SubVeqOp>(operation);
        slice &&
        (!slice.hasConstantLowerBound() || !slice.hasConstantUpperBound())) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool hasBlockingNestedLoop(cudaq::cc::LoopOp loop) {
  bool found = false;
  loop.getBodyRegion().walk([&](cudaq::cc::LoopOp nested) -> WalkResult {
    if (blocksWireConversion(nested)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static uint64_t countClonedOperations(cudaq::cc::LoopOp loop) {
  uint64_t count = 0;
  auto countRegion = [&](Region &region) {
    region.walk([&](Operation *operation) {
      if (!operation->hasTrait<OpTrait::IsTerminator>())
        ++count;
    });
  };
  countRegion(loop.getBodyRegion());
  countRegion(loop.getStepRegion());
  return count;
}

static void cloneAndFold(Operation &source, OpBuilder &builder,
                         IRMapping &mapping) {
  if (auto extract = dyn_cast<cudaq::quake::ExtractRefOp>(source)) {
    Value index = extract.getIndex()
                      ? mapping.lookupOrDefault(extract.getIndex())
                      : Value{};
    if (index)
      if (auto constant = constInt(index); constant && *constant >= 0) {
        auto replacement = cudaq::quake::ExtractRefOp::create(
            builder, extract.getLoc(),
            mapping.lookupOrDefault(extract.getVeq()),
            static_cast<std::size_t>(*constant));
        mapping.map(extract.getResult(), replacement.getResult());
        return;
      }
  }

  if (auto slice = dyn_cast<cudaq::quake::SubVeqOp>(source)) {
    std::optional<int64_t> lower =
        slice.hasConstantLowerBound()
            ? std::optional<int64_t>(slice.getConstantLowerBound())
            : constInt(mapping.lookupOrDefault(slice.getLower()));
    std::optional<int64_t> upper =
        slice.hasConstantUpperBound()
            ? std::optional<int64_t>(slice.getConstantUpperBound())
            : constInt(mapping.lookupOrDefault(slice.getUpper()));
    if (lower && upper && *lower >= 0 && *upper >= 0) {
      auto replacement = cudaq::quake::SubVeqOp::create(
          builder, slice.getLoc(), slice.getType(),
          mapping.lookupOrDefault(slice.getVeq()), Value{}, Value{},
          static_cast<uint64_t>(*lower), static_cast<uint64_t>(*upper));
      mapping.map(slice.getResult(), replacement.getResult());
      return;
    }
  }

  Operation *cloned = builder.clone(source, mapping);
  SmallVector<Value> folded;
  if (succeeded(builder.tryFold(cloned, folded)) && !folded.empty()) {
    assert(folded.size() == source.getNumResults());
    for (auto [original, replacement] :
         llvm::zip_equal(source.getResults(), folded))
      mapping.map(original, replacement);
    cloned->erase();
  }
}

static void cloneStraightLineLoop(cudaq::cc::LoopOp loop,
                                  const FastCountedLoop &counted) {
  OpBuilder builder(loop);
  SmallVector<Value> carried(loop.getInitialArgs());
  Block *body = loop.getDoEntryBlock();
  Block *step = loop.getStepBlock();
  auto bodyContinue = cast<cudaq::cc::ContinueOp>(body->getTerminator());
  auto stepContinue = cast<cudaq::cc::ContinueOp>(step->getTerminator());

  for (uint64_t iteration = 0; iteration < counted.tripCount; ++iteration) {
    IRMapping bodyMapping;
    for (auto [argument, value] :
         llvm::zip_equal(body->getArguments(), carried))
      bodyMapping.map(argument, value);
    for (Operation &operation : body->without_terminator())
      cloneAndFold(operation, builder, bodyMapping);

    SmallVector<Value> afterBody;
    for (Value value : bodyContinue.getOperands())
      afterBody.push_back(bodyMapping.lookupOrDefault(value));

    IRMapping stepMapping;
    for (auto [argument, value] :
         llvm::zip_equal(step->getArguments(), afterBody))
      stepMapping.map(argument, value);
    for (Operation &operation : step->without_terminator())
      cloneAndFold(operation, builder, stepMapping);

    carried.clear();
    for (Value value : stepContinue.getOperands())
      carried.push_back(stepMapping.lookupOrDefault(value));
  }

  loop->replaceAllUsesWith(carried);
  loop.erase();
}

class ExpandQuakeRegisterTraversalsPass
    : public qlx::impl::ExpandQuakeRegisterTraversalsBase<
          ExpandQuakeRegisterTraversalsPass> {
public:
  using ExpandQuakeRegisterTraversalsBase::ExpandQuakeRegisterTraversalsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cudaq::cc::CCDialect, cudaq::quake::QuakeDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> prepared(original.clone());
    if (failed(prepare(*prepared)))
      return signalPassFailure();
    original.getBodyRegion().takeBody(prepared->getBodyRegion());
  }

private:
  LogicalResult prepare(ModuleOp module) {
    uint64_t generated = 0;

    // Post-order collection is the affected-loop queue: inner traversals are
    // handled before parents, while unrelated loops never enter the worklist.
    while (true) {
      SmallVector<std::pair<cudaq::cc::LoopOp, FastCountedLoop>> matched;
      module.walk<WalkOrder::PostOrder>([&](cudaq::cc::LoopOp loop) {
        auto counted = matchFastCountedLoop(loop);
        if (failed(counted) || counted->tripCount > maximumIterations)
          return;
        if (!blocksWireConversion(loop) && !hasBlockingNestedLoop(loop))
          return;
        matched.emplace_back(loop, *counted);
      });

      // Defer an eligible ancestor until eligible descendants are gone. A
      // later round can then observe constants exposed by inner expansion.
      SmallVector<std::pair<cudaq::cc::LoopOp, FastCountedLoop>> candidates;
      for (auto &[loop, counted] : matched) {
        bool hasMatchedDescendant = llvm::any_of(matched, [&](auto &other) {
          return loop != other.first &&
                 loop->isProperAncestor(other.first.getOperation());
        });
        if (hasMatchedDescendant)
          continue;
        uint64_t perIteration = countClonedOperations(loop);
        if (generated > maximumGeneratedOperations ||
            (perIteration &&
             counted.tripCount >
                 (maximumGeneratedOperations - generated) / perIteration))
          continue;
        generated += counted.tripCount * perIteration;
        candidates.emplace_back(loop, counted);
      }
      if (candidates.empty())
        break;

      for (auto &[loop, counted] : candidates)
        cloneStraightLineLoop(loop, counted);
    }

    bool unsupported = false;
    module.walk([&](cudaq::cc::LoopOp loop) {
      if (!blocksWireConversion(loop))
        return;
      unsupported = true;
      auto counted = matchFastCountedLoop(loop);
      if (failed(counted)) {
        loop.emitOpError("wire-blocking register traversal is not a supported "
                         "normalized constant counted loop");
      } else if (counted->tripCount > maximumIterations) {
        loop.emitOpError("wire-blocking register traversal has ")
            << counted->tripCount << " iterations, exceeding the "
            << maximumIterations.getValue() << " preparation limit";
      } else {
        loop.emitOpError("wire-blocking register traversal would exceed the ")
            << maximumGeneratedOperations.getValue()
            << " generated-operation preparation limit";
      }
    });
    return failure(unsupported);
  }
};

} // namespace
