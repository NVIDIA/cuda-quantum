/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
///
/// This file is the internal implementation of "QuakeToQTXConverter."  Most of
/// the heavy lifting required to successfully apply the conversion patterns is
/// done here.  For a brief overview of the conversion process, see the header
/// file corresponding to this file.
///
/// The Quake to QTX conversion is the promotion of qubit references (vector of
/// qubit references) to wires (array of wires), along with the necessary
/// conversion of all operations that use and modify values of these types.
///
/// Example: Take the following Quake operations.
/// ```
///   %q0 = quake.alloca : !quake.qref
///   %q1 = quake.alloca : !quake.qref
///   quake.x [%q0] %q1 : [!quake.qref] !quake.qref
///   quake.z [%q1] %q0 : !quake.qref
/// ```
///
/// Both operations use `%q0` and `%q1`, but each only modifies one of them.
/// (There is a difference between using a qubit reference as a control and as
/// a target.)
///
/// Promoting those qubit references to wires requires converting these
/// operations to QTX:
/// ```
///   %q0_w0 = qtx.alloca : !qtx.wire
///   %q1_w0 = qtx.alloca : !qtx.wire
///   %q1_w1 = qtx.x [%q0_w0] %q1_w0 : [!qtx.wire] !qtx.wire
///      |
///      +--------------+
///                     V
///   %q0_w1 = qtx.z [%q1_w1] %q0_w0 : [!qtx.wire] !qtx.wire
/// ```
///
/// Note the QTX operations consume their target wires and create new values for
/// them.  The same doesn't happen for control wires.  This example shows how
/// Quake quantum values "become" a chain of QTX quantum values:
/// ```
///   Quake            QTX
///    %q0      [%q0_w0 -> %q0_w1]
///    %q1      [%q1_w0 -> %q1_w1]
/// ```
///
/// During conversion, the rewriter does not need to retain those chains
/// explicitly.  It suffices to keep track of the last value in the chain, which
/// is the live wire that corresponds to a qubit reference.  The rewriter uses
/// a map that must be updated whenever an operation consumes a captured value:
/// ```
///   Quake    Initial map  |  After `x`  |  After `z`  |
///   ----------------------|-------------|-------------|
///    %q0      %q0_w0      |             |   %q0_w1    |
///    %q1      %q1_w0      |   %q1_w1    |             |
/// ```
/// (Blank indicating the mapping didn't change)
///
/// This bookkeeping is reasonably straightforward when the kernel contains only
/// one block.  However, when there are multiple blocks (which also happens with
/// operations that have regions), things become complicated:
///
/// Example: An operation with a region.
/// ```
///   %q0 = quake.alloca : !quake.qref
///   %q1 = quake.alloca : !quake.qref
///   foo.bar {
///     quake.x [%q0] %q1 : [!quake.qref] !quake.qref
///   }
///   quake.z [%q1] %q0 : !quake.qref
/// ```
///
/// The conversion, in this case, follows the same principle as before.  The
/// `foo.bar` operation uses and modifies `%q1` and thus needs a new wire
/// for it as a result:
/// ```
///   %q0_w0 = qtx.alloca : !qtx.wire
///   %q1_w0 = qtx.alloca : !qtx.wire
///   %q1_w2 = foo.bar {
///     %q1_w1 = qtx.x [%q0_w0] %q1_w0 : [!qtx.wire] !qtx.wire
///     terminator %q1_w1 : qtx.wire
///   }
///   %q0_w1 = qtx.z [%q1_w2] %q0_w0 : [!qtx.wire] !qtx.wire
/// ```
///
/// Note that after `qtx.x`, both `%q1_w1` and `%q1_w2` are live wires that
/// correspond to the qubit reference `%q0`.  The former, however, is only valid
/// in `foo.bar` region, while the latter is valid in the parent region.  Hence,
/// our tracking of live wires also needs to known in where a live wire is
/// valid.
///
/// Vectors of qubit references, `!quake.qvec`, are treated similarly. Their
/// handling, however, have other caveats because of the following reasons:
///
///   1. QTX's conversion for `quake.qextract`, the `qtx.array_borrow`
///      operation, consumes and create new value for the array that
///      corresponding to the vector.  Hence, when handling operations with
///      regions, we need to track all uses of those values---even if, from
///      Quake perspective, they are not modified:
///      ```
///        foo.bar {
///          %q0 = quake.qextract %qvec[...] : ...
///
///          // After conversion, the `foo.bar` operation will need a result
///          // that is a new value corresponding to `%qvec`
///        }
///      ```
///
///   2. With the exception of `qtx.array_borrow`, QTX's operations can only use
///      arrays that don't have borrowed wires.  This means that it is necessary
///      to yield all wires back to the array before using it---a process that
///      consume these wire values.  Hence the use of an array value implies the
///      use of all wires borrowed from it:
///      ```
///        %q0 = quake.qextract %qvec[...] : ...
///        foo.bar {
///           quake.reset %qvec
///
///          // This region implicitly uses `%q0`.  After conversion, the
///          // `foo.bar` operation will need a result that is a new value
///          // corresponding to `%qvec` and a new value corresponding to `%q0`
///        }
///      ```
///
/// Further details on these caveats and others not mentioned here are described
/// in the relevant parts of the code.
///
#include "QuakeToQTXConverter.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Dialect//ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace cudaq;

#define DEBUG_TYPE "kernel-conversion"

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static bool hasQuantumType(Value value) {
  return value.getType().isa<quake::QRefType>() ||
         value.getType().isa<quake::QVecType>();
};

//===----------------------------------------------------------------------===//
// Kernel analysis
//===----------------------------------------------------------------------===//

// The analysis traverses a Quake kernel in pre-order for two purposes:
//   1. Build a _post_ order list of operations.
//   2. Gather preliminary information about quantum values used by those that
//      have regions.
//
// The reason why operations with regions get special treatment resides in how
// the conversion process works: The rewriter goes through the operation list
// generated by this analysis and has only one shot at converting each by
// applying a conversion pattern.
//
// A pattern must know which quantum values are used, and in the case of
// operation with regions, this would require traversing the regions. Since this
// analysis is visiting the IR recursively, it takes the opportunity to unburden
// the rewriter (or patterns) by gathering the preliminary list of used quantum
// values for these operations.

namespace {

/// The storage class for used quantum values that contains the information
/// necessary to properly infer what a region must return to its parent.
struct UsedQuantumValue {
  Value value;
  bool hasWriteEffect;
};

/// Represents an analysis for computing the list of operations to convert and
/// the quantum values used by those with regions. The analysis recursively
/// iterates over all operations and associated regions.
class KernelAnalysis {
public:
  KernelAnalysis(SmallVectorImpl<Operation *> &toConvert)
      : toConvert(toConvert) {}

  LogicalResult analyze(Operation *op) {
    SmallVector<UsedQuantumValue, 4> usedQuantumValues;
    for (auto &region : op->getRegions())
      if (failed(
              analyze(region.getBlocks(), region.getLoc(), usedQuantumValues)))
        return failure();

    usedQuantumValues.clear();
    if (auto func = dyn_cast<func::FuncOp>(op))
      for (auto arg : func.getArguments())
        if (hasQuantumType(arg))
          usedQuantumValues.push_back({arg, true});
    opUsedQuantumValues.try_emplace(op, usedQuantumValues);
    return success();
  }

  ArrayRef<UsedQuantumValue> getUseQuantumValues(Operation *op) {
    return opUsedQuantumValues[op];
  }

private:
  LogicalResult analyze(iterator_range<Region::iterator> region,
                        Location regionLoc,
                        SmallVectorImpl<UsedQuantumValue> &usedQvalues);

  SmallVectorImpl<Operation *> &toConvert;

  /// The quantum values captured by an operation
  DenseMap<Operation *, SmallVector<UsedQuantumValue, 4>> opUsedQuantumValues;
};
} // namespace

LogicalResult
KernelAnalysis::analyze(iterator_range<Region::iterator> region,
                        Location regionLoc,
                        SmallVectorImpl<UsedQuantumValue> &usedQuantumValues) {
  if (region.empty())
    return success();

  // Traverse starting from the entry block.
  SmallVector<Block *, 16> worklist(1, &*region.begin());
  DenseSet<Block *> visitedBlocks;
  visitedBlocks.insert(worklist.front());
  for (auto i = 0u; i < worklist.size(); ++i) {
    Block *block = worklist[i];

    // Compute the conversion set of each nested operations.
    for (Operation &op : *block) {

      // Search for captured quantum values
      for (auto operand : op.getOperands()) {
        if (operand.getParentRegion() == block->getParent() ||
            !hasQuantumType(operand))
          continue;
        bool hasWriteEffect = hasEffect<MemoryEffects::Write>(&op, operand);
        auto it = llvm::find_if(usedQuantumValues, [&](const auto &entry) {
          return entry.value == operand;
        });
        if (it != usedQuantumValues.end())
          it->hasWriteEffect |= hasWriteEffect;
        else if (hasWriteEffect || operand.getType().isa<quake::QVecType>())
          usedQuantumValues.push_back({operand, hasWriteEffect});
      }

      if (op.getNumRegions() == 0) {
        toConvert.push_back(&op);
        continue;
      }

      // All regions of an operation must return the same set of results.
      SmallVector<UsedQuantumValue, 4> usedValues;
      for (auto &region : op.getRegions()) {
        if (failed(analyze(region.getBlocks(), region.getLoc(), usedValues)))
          return failure();
      }
      if (!usedValues.empty()) {
        for (const auto &value : usedValues) {
          auto it = llvm::find_if(usedQuantumValues, [&](const auto &entry) {
            return entry.value == value.value;
          });
          if (it == usedQuantumValues.end())
            usedQuantumValues.push_back(value);
          else
            it->hasWriteEffect |= value.hasWriteEffect;
        }
        opUsedQuantumValues.try_emplace(&op, usedValues);
      }
      toConvert.push_back(&op);
    }

    // Recurse to children that haven't been visited.
    for (Block *successor : block->getSuccessors())
      if (visitedBlocks.insert(successor).second)
        worklist.push_back(successor);
  }

  // Check that all blocks in the region were visited.
  if (llvm::any_of(llvm::drop_begin(region, 1),
                   [&](Block &block) { return !visitedBlocks.count(&block); }))
    return emitError(regionLoc, "unreachable blocks were not converted");
  return success();
}

//===----------------------------------------------------------------------===//
// OpOriginalState
//===----------------------------------------------------------------------===//

/// The state of an operation that was updated by a pattern in-place, which
/// should only happens when to operations that use the classical result of a
/// measurement operation.  This class contains the necessary information to
/// return an operation to its original state.
class OpOriginalState {
public:
  OpOriginalState() = default;
  OpOriginalState(Operation *op)
      : op(op), operands(op->operand_begin(), op->operand_end()) {}

  /// Discard the transaction state and reset the state of the original
  /// operation.
  void resetOperation() const { op->setOperands(operands); }

  /// Return the original operation of this state.
  Operation *getOperation() const { return op; }

private:
  Operation *op;
  SmallVector<Value, 8> operands;
};

//===----------------------------------------------------------------------===//
// BlockAction
//===----------------------------------------------------------------------===//

/// The kind of the block action performed during the rewrite.  Actions must be
/// undone if the conversion fails.
enum class BlockActionKind { AddArg, Create, Move };

/// Original position of the given block in its parent region. During undo
/// actions, the block needs to be placed after `insertAfterBlock`.
struct BlockPosition {
  Region *region;
  Block *insertAfterBlock;
};

/// The storage class for block action (one of BlockActionKind), which contains
/// the information necessary to undo the action.
struct BlockAction {
  static BlockAction getAddArg(Block *block, unsigned index) {
    BlockAction action = {BlockActionKind::AddArg, block, {}};
    action.argIndex = index;
    return action;
  }

  static BlockAction getCreate(Block *block) {
    return {BlockActionKind::Create, block, {}};
  }

  static BlockAction getMove(Block *block, BlockPosition originalPosition) {
    return {BlockActionKind::Move, block, {originalPosition}};
  }

  BlockActionKind kind;

  /// A pointer to the block that was created/modified by the action.
  Block *block;

  union {
    /// In use if kind == BlockActionKind::Move this contains the information
    /// about its original position
    BlockPosition originalPosition;

    /// In use if kind == BlockActionKind::AddArg this is the index of the arg.
    unsigned argIndex;
  };
};

//===----------------------------------------------------------------------===//
// ConvertToQTXRewriterImpl
//===---------------------------------------------------------------------===//

namespace cudaq::detail {

struct ConvertToQTXRewriterImpl {

  explicit ConvertToQTXRewriterImpl(ConvertToQTXRewriter &rewriter);

  ConvertToQTXRewriterImpl(const ConvertToQTXRewriterImpl &) = delete;
  ConvertToQTXRewriterImpl &
  operator=(const ConvertToQTXRewriterImpl &) = delete;

  /// Cleanup all generated rewrite operations. We need to invoke this method
  /// when the conversion process fails.
  void discardRewrites();

  /// Undo the block actions one by one in reverse order. This is part of the
  /// clean up process triggered by calling `discardRewrites`.
  void undoBlockActions();

  /// Add an argument to a block and create a block action capable of undoing it
  BlockArgument addArgument(Block *block, Type type, Location loc);

  //===--------------------------------------------------------------------===//
  // Mapping
  //===--------------------------------------------------------------------===//

  void mapOrRemap(Value oldValue, Value newValue, Block *block);

  /// Recursively searches for a live wire (array or wires) that corresponds to
  /// a Quake qubit reference (vector of qubit references), `oldValue`.
  Value getRemapped(Operation *op, Value oldValue);

  /// Get the remapped value for all values in the range of `oldValues` while
  /// appending them to `newValues`.
  void getRemapped(Operation *op, ValueRange oldValues,
                   SmallVectorImpl<Value> &newValues);

  Value lookupRecursive(Value oldValue, Block *currentBlock,
                        SmallVectorImpl<std::pair<Block *, Value>> &mappings);

  ArrayRef<UsedQuantumValue> getUsedQuantumValues(Operation *op);

  //===--------------------------------------------------------------------===//
  // Vector helpers
  //===--------------------------------------------------------------------===//

  /// Yield back all borrowed wires that dominate the current array value
  /// corresponding to `qvec`.  Returns a list of qubit references corresponding
  /// to the yield wires.
  ArrayRef<Value> yieldAllWires(Operation *op, Value qvec);

  /// Yield back all borrowed wires that dominate the current array value
  /// corresponding to `qvec` but don't dominate the successors of op.  Returns a
  /// new array value.
  Value yieldPathWires(Operation *op, Value qvec);

  //===--------------------------------------------------------------------===//
  // Rewriter Notification Hooks
  //===--------------------------------------------------------------------===//

  /// Notifies that a block was created.
  void notifyCreatedBlock(Block *block);

  /// Notifies that the blocks of a region are about to be moved.
  void notifyRegionIsBeingInlinedBefore(Region &region, Region &parent,
                                        Region::iterator before);

  /// Notifies that a pattern match failed for the given reason.
  LogicalResult
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback);

  //===--------------------------------------------------------------------===//
  // State
  //===--------------------------------------------------------------------===//

  ConvertToQTXRewriter &rewriter;

  KernelAnalysis *kernelInfo;

  DominanceInfo dominanceInfo;

  /// Ordered vector of all of the newly created operations during conversion.
  SmallVector<Operation *> createdOps;

  /// Ordered vector of all of the erased operations during conversion.
  SmallVector<Operation *> erasedOps;

  /// Mapping of Quake values to QTX values
  DenseMap<Value, SmallVector<std::pair<Block *, Value>, 4>> mappings;

  /// Mapping of Quake values to QTX values
  DenseMap<Value, SmallVector<Value, 4>> extractedQrefs;

  /// Ordered list of block operations (creations, splits, motions).
  SmallVector<BlockAction, 4> blockActions;

  /// The original state for each of operation that was updated in-place.
  SmallVector<OpOriginalState, 4> rootUpdates;

  /// The type converter
  TypeConverter typeConverter;

  /// This allows the user to collect the match failure message.
  function_ref<void(Diagnostic &)> notifyCallback;

#ifndef NDEBUG
  /// A logger used to emit diagnostics during the conversion process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};

ConvertToQTXRewriterImpl::ConvertToQTXRewriterImpl(
    ConvertToQTXRewriter &rewriter)
    : rewriter(rewriter), notifyCallback(nullptr) {
  typeConverter.addConversion([](quake::QRefType type) -> Type {
    return qtx::WireType::get(type.getContext());
  });
  typeConverter.addConversion([](quake::QVecType type) -> Type {
    return type.hasSpecifiedSize()
               ? qtx::WireArrayType::get(type.getContext(), type.getSize(), 0)
               : nullptr;
  });
}

BlockArgument ConvertToQTXRewriterImpl::addArgument(Block *block, Type type,
                                                    Location loc) {
  blockActions.push_back(
      BlockAction::getAddArg(block, block->getNumArguments()));
  return block->addArgument(type, loc);
}

void ConvertToQTXRewriterImpl::mapOrRemap(Value from, Value to,
                                          Block *toBlock) {
  auto &valueMappings = mappings[from];
  for (auto &[block, value] : llvm::reverse(valueMappings)) {
    if (toBlock == block) {
      value = to;
      return;
    }
  }
  valueMappings.emplace_back(toBlock, to);
}

ArrayRef<UsedQuantumValue>
ConvertToQTXRewriterImpl::getUsedQuantumValues(Operation *op) {
  return kernelInfo->getUseQuantumValues(op);
}

Value ConvertToQTXRewriterImpl::getRemapped(Operation *op, Value oldValue) {
  auto it = mappings.find(oldValue);
  if (it == mappings.end())
    return oldValue; // non-quantum value
  auto remapped = lookupRecursive(oldValue, op->getBlock(), it->second);
  assert(remapped && "Failed to remap value");
  return remapped;
}

void ConvertToQTXRewriterImpl::getRemapped(Operation *op, ValueRange oldValues,
                                           SmallVectorImpl<Value> &newValues) {
  llvm::transform(oldValues, std::back_inserter(newValues),
                  [&](Value value) { return getRemapped(op, value); });
}

static Value yieldWires(Operation *op, Value qvec, ArrayRef<Value> qrefs,
                        ConvertToQTXRewriter &rewriter) {
  // Get the corresponding array
  Value array = rewriter.getRemapped(op, qvec);

  // Get the corresponding wires
  SmallVector<Value, 4> wires;
  rewriter.getRemapped(op, qrefs, wires);

  // Yield the wires back and update the vector mapping to the new array
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  array = rewriter.create<qtx::ArrayYieldOp>(op->getLoc(), wires, array);
  rewriter.mapOrRemap(qvec, array);

  return array;
}

ArrayRef<Value> ConvertToQTXRewriterImpl::yieldAllWires(Operation *op,
                                                        Value qvec) {
  using Iterator = SmallVector<Value>::iterator;

  auto &extractedQrefs = rewriter.getImpl().extractedQrefs[qvec];
  if (extractedQrefs.empty())
    return {};

  Iterator it = llvm::partition(extractedQrefs, [&](Value qref) {
    return rewriter.getImpl().dominanceInfo.dominates(qref, op);
  });

  if (it == extractedQrefs.begin())
    return {};

  ArrayRef<Value> qrefs(extractedQrefs.begin(), it);
  yieldWires(op, qvec, qrefs, rewriter);
  return qrefs;
}

Value ConvertToQTXRewriterImpl::yieldPathWires(Operation *op, Value qvec) {
  using Iterator = SmallVector<Value>::iterator;

  auto &extractedQrefs = rewriter.getImpl().extractedQrefs[qvec];
  if (extractedQrefs.empty())
    return nullptr;

  Iterator it = llvm::partition(extractedQrefs, [&](Value qref) {
    auto parentBlock = qref.getParentBlock();
    if (!rewriter.getImpl().dominanceInfo.dominates(qref, op))
      return true;
    if (!op->hasSuccessors() && op->getParentRegion() == qref.getParentRegion())
      return false;
    for (auto succ : op->getSuccessors())
      if (!rewriter.getImpl().dominanceInfo.dominates(parentBlock, succ))
        return false;
    return true;
  });

  if (it == extractedQrefs.end())
    return nullptr;

  ArrayRef<Value> qrefs(it, extractedQrefs.end());
  return yieldWires(op, qvec, qrefs, rewriter);
}

Value ConvertToQTXRewriterImpl::lookupRecursive(
    Value oldValue, Block *currentBlock,
    SmallVectorImpl<std::pair<Block *, Value>> &mappings) {
  // Search for a mapping valid in the current block.
  for (auto &[block, value] : llvm::reverse(mappings)) {
    if (block == currentBlock)
      return value;
  }

  // If the current block does not have a predecessor, then we need to look for
  // a mapping in the enclosing region, i.e., the region where this block's
  // parent lives.
  if (currentBlock->hasNoPredecessors()) {
    auto op = currentBlock->getParentOp();
    assert(op && !op->hasTrait<OpTrait::IsIsolatedFromAbove>());
    auto result = lookupRecursive(oldValue, op->getBlock(), mappings);
    mappings.emplace_back(currentBlock, result);
    return result;
  }

  // If the current block has one predecessor, search for the mapping there.
  if (auto pred = currentBlock->getUniquePredecessor()) {
    auto result = lookupRecursive(oldValue, pred, mappings);
    mappings.emplace_back(currentBlock, result);
    return result;
  }

  // The current block has multiple predecessors, and thus we collect valid
  // mappings from all of them.  Note, however, that this search might lead to
  // further recursive lookups---which might lead to endless recursion due to
  // the presence of loops.  Hence, before recursing, we create an invalid
  // mapping for the current block:
  auto &map = mappings.emplace_back(currentBlock, nullptr);
  SmallVector<Block *, 4> predecessors(currentBlock->getPredecessors());
  SmallVector<Value, 4> predMappings;
  for (auto p : predecessors) {
    auto result = lookupRecursive(oldValue, p, mappings);
    predMappings.push_back(result);
  }

  if (llvm::all_equal(predMappings))
    return map.second = predMappings.front();

  // The predecessor mappings are not the same, and thus we will need to add
  // a new argument to the block.  First, we need to figure out its type.
  Type type = oldValue.getType().isa<quake::QRefType>()
                  ? qtx::WireType::get(oldValue.getContext())
                  : nullptr;

  // If `oldValue` is a vector of qubit references, we need to iterate over its
  // predecessor mappings and yield back all extracted references that do not
  // dominate this block.  (Note a predecessor mapping can be a invalid value,
  // which indicates a cycle.)
  if (!type)
    for (auto &&[value, p] : llvm::zip_equal(predMappings, predecessors)) {
      if (!value)
        continue;
      auto new_array = yieldPathWires(p->getTerminator(), oldValue);
      if (new_array)
        value = new_array;
      type = value.getType();
    }

  map.second = addArgument(currentBlock, type, oldValue.getLoc());
  for (auto [value, p] : llvm::zip_equal(predMappings, predecessors)) {
    if (!value)
      value = map.second;

    auto term = p->getTerminator();
    // TODO: Look for some interface that allows dealing with terminators in an
    // abstract way.
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      br.getDestOperandsMutable().append(value);
    } else if (auto cond_br = dyn_cast<cf::CondBranchOp>(term)) {
      auto operands = currentBlock == cond_br.getTrueDest()
                         ? cond_br.getTrueDestOperandsMutable()
                         : cond_br.getFalseDestOperandsMutable();
      operands.append(value);
    }
  }
  return map.second;
}

void ConvertToQTXRewriterImpl::notifyCreatedBlock(Block *block) {
  blockActions.push_back(BlockAction::getCreate(block));
}

void ConvertToQTXRewriterImpl::notifyRegionIsBeingInlinedBefore(
    Region &region, Region &parent, Region::iterator before) {
  if (region.empty())
    return;
  Block *laterBlock = &region.back();
  for (auto &earlierBlock : llvm::drop_begin(llvm::reverse(region), 1)) {
    blockActions.push_back(
        BlockAction::getMove(laterBlock, {&region, &earlierBlock}));
    laterBlock = &earlierBlock;
  }
  blockActions.push_back(BlockAction::getMove(laterBlock, {&region, nullptr}));
}

LogicalResult ConvertToQTXRewriterImpl::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  LLVM_DEBUG({
    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);
    logger.startLine() << "** Failure : " << diag.str() << "\n";
    if (notifyCallback)
      notifyCallback(diag);
  });
  return failure();
}

void ConvertToQTXRewriterImpl::discardRewrites() {
  // Reset any operations that were updated in place.
  for (auto &state : rootUpdates)
    state.resetOperation();

  undoBlockActions();

  // Remove any newly created ops.
  for (auto *op : llvm::reverse(createdOps)) {
    op->dropAllUses();
    op->erase();
  }
}

void ConvertToQTXRewriterImpl::undoBlockActions() {
  for (auto &action : llvm::reverse(blockActions)) {
    switch (action.kind) {
    // Delete the created block arguments.
    case BlockActionKind::AddArg: {
      BlockArgument arg = action.block->getArgument(action.argIndex);
      arg.dropAllUses();
      action.block->eraseArgument(action.argIndex);
      break;
    }
    // Delete the created block.
    case BlockActionKind::Create: {
      // Unlink all of the operations within this block, they will be deleted
      // separately.
      auto &blockOps = action.block->getOperations();
      while (!blockOps.empty())
        blockOps.remove(blockOps.begin());
      action.block->dropAllDefinedValueUses();
      action.block->erase();
      break;
    }
    // Move the block back to its original position.
    case BlockActionKind::Move: {
      Region *originalRegion = action.originalPosition.region;
      Block *insertAfterBlock = action.originalPosition.insertAfterBlock;
      originalRegion->getBlocks().splice(
          (insertAfterBlock ? std::next(Region::iterator(insertAfterBlock))
                            : originalRegion->end()),
          action.block->getParent()->getBlocks(), action.block);
      break;
    }
    }
  }
}
} // namespace cudaq::detail

//===----------------------------------------------------------------------===//
// ConvertToQTXRewriter
//===----------------------------------------------------------------------===//

ConvertToQTXRewriter::ConvertToQTXRewriter(MLIRContext *ctx)
    : PatternRewriter(ctx), impl(new detail::ConvertToQTXRewriterImpl(*this)) {}

ConvertToQTXRewriter::~ConvertToQTXRewriter() = default;

//===----------------------------------------------------------------------===//
// Hooks

void ConvertToQTXRewriter::notifyOperationInserted(Operation *op) {
  LLVM_DEBUG({
    impl->logger.startLine()
        << "** Insert  : '" << op->getName() << "'(" << op << ")\n";
  });
  impl->createdOps.push_back(op);
}

void ConvertToQTXRewriter::eraseOp(Operation *op) {
  LLVM_DEBUG({
    impl->logger.startLine()
        << "** Erase   : '" << op->getName() << "'(" << op << ")\n";
  });
  impl->erasedOps.push_back(op);
}

BlockArgument ConvertToQTXRewriter::addArgument(Block *block, Type type,
                                                Location loc) {
  return impl->addArgument(block, type, loc);
}

void ConvertToQTXRewriter::notifyBlockCreated(Block *block) {
  impl->notifyCreatedBlock(block);
}

void ConvertToQTXRewriter::inlineRegionBefore(Region &region, Region &parent,
                                              Region::iterator before) {
  impl->notifyRegionIsBeingInlinedBefore(region, parent, before);
  PatternRewriter::inlineRegionBefore(region, parent, before);
}

void ConvertToQTXRewriter::startRootUpdate(Operation *op) {
  auto &rootUpdates = impl->rootUpdates;
  auto it = llvm::find_if(llvm::reverse(rootUpdates), [op](const auto &it) {
    return it.getOperation() == op;
  });
  if (it != rootUpdates.rend())
    return;
  rootUpdates.emplace_back(op);
}

void ConvertToQTXRewriter::cancelRootUpdate(Operation *op) {
  auto &rootUpdates = impl->rootUpdates;
  auto it = llvm::find_if(llvm::reverse(rootUpdates), [op](const auto &it) {
    return it.getOperation() == op;
  });
  assert(it != rootUpdates.rend() && "no root update started on op");
  (*it).resetOperation();
  int updateIdx = std::prev(rootUpdates.rend()) - it;
  rootUpdates.erase(rootUpdates.begin() + updateIdx);
}

//===----------------------------------------------------------------------===//
// Value (re)mapping and type conversion

Type ConvertToQTXRewriter::convertType(mlir::Type type) {
  return impl->typeConverter.convertType(type);
}

void ConvertToQTXRewriter::mapOrRemap(Value from, Value to) {
  auto &blockMappings = impl->mappings[from];
  auto toDefBlock = to.getParentBlock();
  for (auto &[block, value] : llvm::reverse(blockMappings)) {
    if (toDefBlock == block) {
      value = to;
      return;
    }
  }
  blockMappings.emplace_back(toDefBlock, to);
}

Value ConvertToQTXRewriter::getRemapped(Operation *op, Value oldValue) {
  return impl->getRemapped(op, oldValue);
}

void ConvertToQTXRewriter::getRemapped(Operation *op, ValueRange oldValues,
                                       SmallVectorImpl<Value> &remapped) {
  impl->getRemapped(op, oldValues, remapped);
}

cudaq::detail::ConvertToQTXRewriterImpl &ConvertToQTXRewriter::getImpl() {
  return *impl;
}

//===----------------------------------------------------------------------===//
// CircuitRewritePattern
//===----------------------------------------------------------------------===//

struct WireReborrower {
  WireReborrower(Operation *op, Value qvec, ArrayRef<Value> extracted,
                 ConvertToQTXRewriter &rewriter)
      : op(op), qvec(qvec), extracted(extracted), rewriter(rewriter) {}

  ~WireReborrower() {
    assert(!op->hasTrait<OpTrait::IsTerminator>() &&
           "Cannot re-borrow wires after a terminator");
    SmallVector<Value, 4> indices;
    for (auto qref : extracted) {
      auto extractOp = dyn_cast<quake::QExtractOp>(qref.getDefiningOp());
      indices.push_back(extractOp.getIndex());
    }
    Value array = rewriter.getRemapped(op, qvec);
    auto borrow =
        rewriter.create<qtx::ArrayBorrowOp>(op->getLoc(), indices, array);
    for (auto [qref, wire] : llvm::zip_equal(extracted, borrow.getWires()))
      rewriter.mapOrRemap(qref, wire);
    rewriter.mapOrRemap(qvec, borrow.getNewArray());
  }

  Operation *op;
  Value qvec;
  ArrayRef<Value> extracted;
  ConvertToQTXRewriter &rewriter;
};

static LogicalResult
computeTerminatorUsedQuantumValues(Operation *op,
                                   SmallVectorImpl<Value> &usedQuantumValues,
                                   ConvertToQTXRewriter &rewriter) {
  auto &rewriterImpl = rewriter.getImpl();
  // If this operation is a terminator, we need to get the parent's list of
  // used quantum values because those will become the operands for the new
  // terminator.
  for (auto &useInfo : rewriterImpl.getUsedQuantumValues(op->getParentOp())) {
    auto value = useInfo.value;
    usedQuantumValues.push_back(value);
    if (!value.getType().isa<quake::QVecType>())
      continue;
    rewriterImpl.yieldPathWires(op, value);
    if (useInfo.hasWriteEffect)
      llvm::copy_if(rewriterImpl.extractedQrefs[value],
                    std::back_inserter(usedQuantumValues), [&](Value qref) {
                      return rewriterImpl.dominanceInfo.dominates(qref, op) &&
                             qref.getParentRegion() != op->getParentRegion();
                    });
  }
  return success();
}

static LogicalResult
computeOpRegionUsedQuantumValues(Operation *op,
                                 SmallVectorImpl<Value> &usedQuantumValues,
                                 ConvertToQTXRewriter &rewriter) {
  auto &rewriterImpl = rewriter.getImpl();

  // Get the preliminary set of used quantum values and extend it with
  // implicitly used ones that dominate this op---which happens when we use a
  // quantum vector with an operation that has a writing effect to it.
  for (const auto &useInfo : rewriterImpl.getUsedQuantumValues(op)) {
    auto value = useInfo.value;
    usedQuantumValues.push_back(value);
    if (value.getType().isa<quake::QVecType>() && useInfo.hasWriteEffect) {
      llvm::copy_if(rewriterImpl.extractedQrefs[value],
                    std::back_inserter(usedQuantumValues), [&](Value qref) {
                      return rewriterImpl.dominanceInfo.dominates(qref, op);
                    });
    }
  }

  // Get the live QTX values that correspond to Quake's quantum values.
  SmallVector<Value, 4> remapped;
  rewriterImpl.getRemapped(op, usedQuantumValues, remapped);
  return success();
}

LogicalResult
cudaq::ConvertToQTXPattern::matchAndRewrite(Operation *op,
                                            PatternRewriter &rewriter) const {
  auto &conversionRewriter = static_cast<ConvertToQTXRewriter &>(rewriter);
  auto &rewriterImpl = conversionRewriter.getImpl();

  SmallVector<Value, 4> usedQuantumValues;
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    if (failed(computeTerminatorUsedQuantumValues(op, usedQuantumValues,
                                                  conversionRewriter)))
      return failure();
  } else if (op->getNumRegions() > 0) {
    if (failed(computeOpRegionUsedQuantumValues(op, usedQuantumValues,
                                                conversionRewriter)))
      return failure();
    // If there are no qubit usage inside the region, we don't need to convert
    // this operation.
    if (usedQuantumValues.empty())
      return success();
  }

  // Handle special cases of using quantum vector as operands
  SmallVector<WireReborrower, 4> borrowers;
  if (auto qextract = dyn_cast<quake::QExtractOp>(op)) {
    auto &extracted = rewriterImpl.extractedQrefs[qextract.getQvec()];
    extracted.push_back(qextract.getQref());
  } else if (auto dealloc = dyn_cast<quake::DeallocOp>(op)) {
    auto qrefOrQvec = dealloc.getQregOrVec();
    if (qrefOrQvec.getType().isa<quake::QVecType>())
      // Okay to discard as we don't need to borrow wires again.
      (void)rewriterImpl.yieldAllWires(op, qrefOrQvec);
  } else {
    for (auto operand : op->getOperands()) {
      if (operand.getType().isa<quake::QVecType>()) {
        auto extracted = rewriterImpl.yieldAllWires(op, operand);
        if (!extracted.empty())
          borrowers.emplace_back(op, operand, extracted, conversionRewriter);
      }
    }
  }

  // Remap the operands of the operation.
  SmallVector<Value, 4> operands;
  rewriterImpl.getRemapped(op, op->getOperands(), operands);

  return matchAndRewrite(op, usedQuantumValues, operands, conversionRewriter);
}

//===----------------------------------------------------------------------===//
// OperationConverter
//===----------------------------------------------------------------------===//

namespace {
/// This class defines a operation converter.
class OperationConverter {
public:
  OperationConverter(const FrozenRewritePatternSet &patterns);

  /// Attempt to convert the given operation. Returns success if the operation
  /// was converted, failure otherwise.
  LogicalResult convert(Operation *op, cudaq::ConvertToQTXRewriter &rewriter);

private:
  /// Attempt to convert the given operation by applying a pattern. Returns
  /// success if the operation was converted, failure otherwise.
  LogicalResult convertWithPattern(Operation *op,
                                   cudaq::ConvertToQTXRewriter &rewriter);

  /// The pattern applicator to use for conversions.
  PatternApplicator applicator;
};
} // namespace

OperationConverter::OperationConverter(const FrozenRewritePatternSet &patterns)
    : applicator(patterns) {
  applicator.applyDefaultCostModel();
}

LogicalResult
OperationConverter::convert(Operation *op,
                            cudaq::ConvertToQTXRewriter &rewriter) {
  auto &rewriterImp = rewriter.getImpl();
#ifndef NDEBUG
  const char *logSeparator =
      "//===-------------------------------------------===//\n";

  auto &logger = rewriterImp.logger;
#endif
  LLVM_DEBUG({
    logger.getOStream() << "\n";
    logger.startLine() << logSeparator;
    logger.startLine() << "Converting operation : '" << op->getName() << "'("
                       << op << ") {\n";
    logger.indent();

    if (op->getNumRegions() == 0) {
      op->print(logger.startLine(), OpPrintingFlags().printGenericOpForm());
      logger.getOStream() << "\n\n";
    }
  });

  rewriter.setInsertionPoint(op);
  if (succeeded(convertWithPattern(op, rewriter))) {
    LLVM_DEBUG({
      logger.unindent();
      logger.startLine() << "} -> SUCCESS\n";
      logger.startLine() << logSeparator;
    });
    return success();
  }

  LLVM_DEBUG({
    logger.unindent();
    logger.startLine() << "} -> FAILURE : unrecognised operation \n";
    logger.startLine() << logSeparator;
  });

  // This should fail if we miss a quake operation or if an operation requires
  // results but was not converted.
  if (isa<quake::QuakeDialect>(op->getDialect()) ||
      !rewriterImp.getUsedQuantumValues(op).empty())
    return failure();

  return success();
}

LogicalResult
OperationConverter::convertWithPattern(Operation *op,
                                       ConvertToQTXRewriter &rewriter) {
  auto &rewriterImpl = rewriter.getImpl();

  auto canApply = [&](const Pattern &pattern) { return true; };

  auto onFailure = [&](const Pattern &pattern) {
    LLVM_DEBUG({
      rewriterImpl.logger.startLine()
          << "} -> FAILURE : pattern failed to match\n";
      if (rewriterImpl.notifyCallback) {
        Diagnostic diag(op->getLoc(), DiagnosticSeverity::Remark);
        diag << "Failed to apply pattern \"" << pattern.getDebugName()
             << "\" on op:\n"
             << *op;
        rewriterImpl.notifyCallback(diag);
      }
    });
  };

  return applicator.matchAndRewrite(op, rewriter, canApply, onFailure);
}

//===----------------------------------------------------------------------===//
// ConversionDriver
//===----------------------------------------------------------------------===//

LogicalResult cudaq::applyPartialQuakeToQTXConversion(
    Operation *op, const FrozenRewritePatternSet &patterns,
    function_ref<void(Diagnostic &)> notifyCallback) {
  // Collect the list of operations to convert.
  SmallVector<Operation *> toConvert;
  KernelAnalysis kernelInfo(toConvert);

  if (failed(kernelInfo.analyze(op)))
    return failure();

  ConvertToQTXRewriter rewriter(op->getContext());
  auto &rewriterImpl = rewriter.getImpl();
  rewriterImpl.kernelInfo = &kernelInfo;
  rewriterImpl.notifyCallback = notifyCallback;

  // Convert entry block arguments
  for (auto &region : op->getRegions()) {
    if (region.empty())
      continue;
    auto entryBlock = &*region.begin();
    rewriter.setInsertionPointToStart(entryBlock);
    for (auto arg : entryBlock->getArguments()) {
      if (!hasQuantumType(arg))
        continue;
      auto desiredType = rewriter.convertType(arg.getType());
      if (!desiredType) {
        rewriter.getImpl().discardRewrites();
        return failure();
      }
      auto convertOp = rewriter.create<UnrealizedConversionCastOp>(
          arg.getLoc(), desiredType, arg);
      rewriter.mapOrRemap(arg, convertOp.getResult(0));
    }
  }

  // Convert each operation and discard rewrites on failure.
  OperationConverter converter(patterns);
  for (auto *op : toConvert)
    if (failed(converter.convert(op, rewriter))) {
      rewriter.getImpl().discardRewrites();
      return failure();
    }

  for (auto op : llvm::reverse(rewriter.getImpl().erasedOps))
    op->erase();

  return success();
}
