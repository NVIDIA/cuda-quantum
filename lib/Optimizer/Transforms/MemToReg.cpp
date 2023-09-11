/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <deque>

namespace cudaq::opt {
#define GEN_PASS_DEF_MEMTOREG
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "memtoreg"

using namespace mlir;

static bool isMemoryAlloc(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op))
    return iface.hasEffect<MemoryEffects::Allocate>();
  return false;
}

static bool isMemoryUse(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op))
    return iface.hasEffect<MemoryEffects::Read>();
  return false;
}

static bool isMemoryDef(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op))
    return iface.hasEffect<MemoryEffects::Write>();
  return false;
}

static bool isFunctionEntryBlock(Block *block) {
  if (isa<func::FuncOp, cudaq::cc::CreateLambdaOp>(block->getParentOp()))
    return block->isEntryBlock();
  return false;
}

namespace {
/// Determine the allocations in this scope (a function) whose lifetime is
/// limited to the scope and which do not escape the scope.
struct MemoryAnalysis {
  MemoryAnalysis(func::FuncOp f) { determineAllocSet(f); }

  bool isMember(Operation *op) const { return allocSet.count(op); }

  SmallVector<quake::AllocaOp> getAllQuantumAllocations() const {
    SmallVector<quake::AllocaOp> result;
    for (auto *op : allocSet)
      if (auto qalloc = dyn_cast<quake::AllocaOp>(op))
        result.push_back(qalloc);
    return result;
  }

private:
  void determineAllocSet(func::FuncOp func) {
    SmallVector<Operation *> allocations;
    auto qrefTy = quake::RefType::get(func.getContext());
    func->walk([&](Operation *op) {
      if (isMemoryAlloc(op)) {
        // Make sure this is stack here. Can we make use of an Interface?
        if (auto alloc = dyn_cast<quake::AllocaOp>(op)) {
          if (alloc.getType() == qrefTy)
            allocations.push_back(op);
        } else if (auto alloc = dyn_cast<cudaq::cc::AllocaOp>(op)) {
          if (!alloc.getSeqSize()) {
            LLVM_DEBUG(llvm::dbgs() << "adding: " << alloc << '\n');
            allocations.push_back(op);
          }
        }
      }
    });
    for (auto *a : allocations) {
      auto *add = a;
      for (auto *u : a->getUsers())
        if (!isMemoryUse(u) && !isMemoryDef(u)) {
          add = nullptr;
          break;
        }
      if (add)
        allocSet.insert(add);
    }
  }

  SmallPtrSet<Operation *, 4> allocSet;
};
} // namespace

/// Generic traversal over an operation, \p op, that collects all its exit
/// blocks. If the operation does not have regions, an empty deque is returned.
/// Otherwise, the exit blocks are for all regions in \p op are returned.
static std::deque<Block *> collectAllExits(Operation *op) {
  std::deque<Block *> blocks;
  for (auto &region : op->getRegions())
    for (auto &block : region)
      if (block.hasNoSuccessors())
        blocks.push_back(&block);
  return blocks;
}

static void
appendPredecessorsToWorklist(std::deque<Block *> &worklist, Block *block,
                             const SmallPtrSetImpl<Block *> &blocksVisited) {
  if (block->hasNoPredecessors())
    return;
  for (auto *p : block->getPredecessors())
    if (!blocksVisited.count(p))
      worklist.push_back(p);
}

static bool opResultOfType(Operation *op, Type ofTy) {
  auto results = op->getResults();
  for (auto r : results)
    if (r.getType() == ofTy)
      return true;
  return false;
}

/// Return true if and only if the value \p defVal is the result of an Operation
/// owned by the operation \p op.
static bool isDescendantOf(Operation *op, Value defVal) {
  if (auto *def = defVal.getDefiningOp())
    return op->isAncestor(def);
  for (auto *parent = cast<BlockArgument>(defVal).getOwner()->getParentOp();
       parent; parent = parent->getParentOp())
    if (parent == op)
      return true;
  return false;
}

// FIXME: llvm::MapVector is understood to be a heavyweight container, but these
// maps ought to be quite small in size. The number of exit blocks in an op with
// regions (typically 1 or 2) * the number of distinct qubits (on the order of
// ten, at most).
using RegionDefinitionsMap =
    llvm::MapVector<Block *, llvm::MapVector<Value, Value>>;

static SmallVector<Value>
getAllDefinitions(const RegionDefinitionsMap &defMap) {
  SetVector<Value> results;
  for (auto &[block, defs] : defMap)
    for (auto &[defKey, localVal] : defs)
      results.insert(defKey);
  return {results.begin(), results.end()};
}

namespace {
/// The reset operation is a bit of an oddball and doesn't support the
/// QuakeOperator interface. Handle it special for now.
class ResetOpPattern : public OpRewritePattern<quake::ResetOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ResetOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto wireTy = quake::WireType::get(rewriter.getContext());
    auto opnd = op.getTargets();
    assert(opnd.getType() == quake::RefType::get(rewriter.getContext()));
    Value target = rewriter.create<quake::UnwrapOp>(loc, wireTy, opnd);
    auto newOp =
        rewriter.create<quake::ResetOp>(loc, TypeRange{wireTy}, target);
    rewriter.replaceOpWithNewOp<quake::WrapOp>(op, newOp.getResult(0), opnd);
    return success();
  }
};

class DeallocOpPattern : public OpRewritePattern<quake::DeallocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto wireTy = quake::WireType::get(rewriter.getContext());
    auto opnd = op.getReference();
    assert(isa<quake::RefType>(opnd.getType()));
    Value target = rewriter.create<quake::UnwrapOp>(loc, wireTy, opnd);
    rewriter.replaceOpWithNewOp<quake::SinkOp>(op, target);
    return success();
  }
};
} // namespace

template <typename OP>
class Wrapper : public OpRewritePattern<OP> {
public:
  using Base = OpRewritePattern<OP>;
  using Base::Base;

  LogicalResult matchAndRewrite(OP op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value> unwrapCtrls;
    auto wireTy = quake::WireType::get(rewriter.getContext());
    auto qrefTy = quake::RefType::get(rewriter.getContext());
    // Scan the control and target positions. Any that were not Wires will be
    // promoted to Wires via an unwrap operation. These unwrap ops become the
    // arguments to the quantum value form of the new quantum operation.
    if constexpr (!quake::isMeasure<OP>) {
      for (auto opnd : op.getControls()) {
        auto opndTy = opnd.getType();
        if (opndTy == qrefTy) {
          auto unwrap = rewriter.create<quake::UnwrapOp>(loc, wireTy, opnd);
          unwrapCtrls.push_back(unwrap);
        } else {
          unwrapCtrls.push_back(opnd);
        }
      }
    }
    SmallVector<Value> unwrapTargs;
    for (auto opnd : op.getTargets()) {
      auto opndTy = opnd.getType();
      if (opndTy == qrefTy) {
        auto unwrap = rewriter.create<quake::UnwrapOp>(loc, wireTy, opnd);
        unwrapTargs.push_back(unwrap);
      } else {
        unwrapTargs.push_back(opnd);
      }
    }
    if constexpr (quake::isMeasure<OP>) {
      // The result type of the bits is the same. Add the wire types.
      SmallVector<Type> newTy = {op.getBits().getType()};
      SmallVector<Type> wireTys(unwrapTargs.size(), wireTy);
      newTy.append(wireTys.begin(), wireTys.end());
      rewriter.replaceOpWithNewOp<OP>(op, newTy, unwrapTargs,
                                      op.getRegisterNameAttr());
    } else {
      // Scan the control and target positions. Any that were not wires
      // originally are now placed in the result vector. Those new results are
      // propagated to wrap operations.
      auto numberOfWires = unwrapCtrls.size() + unwrapTargs.size();
      SmallVector<Type> wireTys{numberOfWires, wireTy};
      auto newOp = rewriter.create<OP>(
          loc, wireTys, op.getIsAdjAttr(), op.getParameters(), unwrapCtrls,
          unwrapTargs, op.getNegatedQubitControlsAttr());
      SmallVector<Value> wireOperands = op.getControls();
      wireOperands.append(op.getTargets().begin(), op.getTargets().end());
      for (auto i : llvm::enumerate(wireOperands)) {
        auto opndTy = i.value().getType();
        unsigned count = 0;
        if (opndTy == qrefTy) {
          rewriter.create<quake::WrapOp>(loc, newOp.getResult(i.index()),
                                         i.value());
        } else if (opndTy == wireTy) {
          op.getResult(count++).replaceAllUsesWith(newOp.getResult(i.index()));
        }
      }
      rewriter.eraseOp(op);
    }
    return success();
  }
};

static Type unwrapType(Type ty) {
  if (isa<quake::RefType>(ty))
    return quake::WireType::get(ty.getContext());
  return cast<cudaq::cc::PointerType>(ty).getElementType();
}

#define WRAPPER(OpClass) Wrapper<quake::OpClass>
#define WRAPPER_QUANTUM_OPS QUANTUM_OPS(WRAPPER)
#define RAW(OpClass) quake::OpClass
#define RAW_QUANTUM_OPS QUANTUM_OPS(RAW)

namespace {
class MemToRegPass : public cudaq::opt::impl::MemToRegBase<MemToRegPass> {
public:
  using MemToRegBase::MemToRegBase;
  using DefnMap = DenseMap<Value, Value>;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Function before memtoreg:\n" << func << "\n\n");

    if (!quantumValues && !classicalValues) {
      // nothing to do
      LLVM_DEBUG(llvm::dbgs() << "memtoreg: both quantum and classical "
                                 "transformations are disabled.\n");
      return;
    }

    // 1) Rewrite the quantum operations into the intermediate QLS form.
    if (failed(convertToQLS()))
      return;

    // 2) Convert load/store memory ops to value form.
    MemoryAnalysis memAnalysis(func);
    SmallPtrSet<Operation *, 4> cleanUps;
    processOpWithRegions(func, memAnalysis, cleanUps);

    // 3) Cleanup the dead ops.
    SmallVector<quake::WrapOp> wrapOps;
    for (auto *op : cleanUps) {
      if (auto wrap = dyn_cast<quake::WrapOp>(op)) {
        wrapOps.push_back(wrap);
        continue;
      }
      op->dropAllUses();
      op->erase();
    }
    for (auto wrap : wrapOps) {
      auto ref = wrap.getRefValue();
      auto wire = wrap.getWireValue();
      if (!ref || wire.getUses().empty()) {
        wrap->dropAllUses();
        wrap->erase();
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Finalized:\n" << func << "\n\n");
  }

  void handleSubRegions(Operation *parent, const MemoryAnalysis &memAnalysis,
                        SmallPtrSetImpl<Operation *> &cleanUps) {
    for (auto &region : parent->getRegions())
      for (auto &block : region)
        for (auto &op : block)
          if (op.getNumRegions())
            processOpWithRegions(&op, memAnalysis, cleanUps);
  }

  void processOpWithRegions(Operation *parent,
                            const MemoryAnalysis &memAnalysis,
                            SmallPtrSetImpl<Operation *> &cleanUps) {
    auto *ctx = &getContext();
    auto wireTy = quake::WireType::get(ctx);
    auto qrefTy = quake::RefType::get(ctx);

    if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(parent)) {
      // Special case: add an else region if it is absent from parent.
      auto &elseRegion = ifOp.getElseRegion();
      if (elseRegion.empty()) {
        auto block = new Block;
        elseRegion.push_back(block);
        OpBuilder builder(ctx);
        builder.setInsertionPointToEnd(block);
        builder.create<cudaq::cc::ContinueOp>(ifOp.getLoc());
      }
    }

    // defMap is the persistent definition map used to determine the origin of a
    // quantum reference value.
    DenseMap<Block *, DefnMap> defMap;

    // First, if any operations held by the blocks of \p parent contain regions,
    // recursively process those operations. This establishes the value
    // semantics interface for these macro ops.
    handleSubRegions(parent, memAnalysis, cleanUps);

    // \p parent is either a callable computation or an inlined macro
    // computation. In the former case, the signature of the call will not be
    // changed. In the latter case, \p parent will be updated to thread through
    // any values that are promoted because of possible modification in the
    // regions of \p parent.
    bool isCallableOp = isa<func::FuncOp, cudaq::cc::CreateLambdaOp>(parent);

    DenseMap<Block *, SmallVector<Value>> blockArgsMap;
    SmallPtrSet<Block *, 4> blocksVisited;
    RegionDefinitionsMap regionDefs;
    DefnMap promotedDefs;
    auto worklist = collectAllExits(parent);
    while (!worklist.empty()) {
      Block *block = worklist.front();
      worklist.pop_front();
      blocksVisited.insert(block);
      // valMap is the per iteration on a block value map: for the current block
      // forward defs (stores) to uses (load users).
      DefnMap valMap;
      if (defMap.count(block))
        valMap = defMap[block];
      blockArgsMap.insert({block, SmallVector<Value>{}});
      regionDefs.insert({block, llvm::MapVector<Value, Value>{}});

      // If this is the entry block and there are quantum reference arguments
      // into the function, promote them to wire values immediately.
      if (quantumValues && isFunctionEntryBlock(block)) {
        for (auto arg : block->getArguments()) {
          if (arg.getType() == qrefTy) {
            OpBuilder builder(ctx);
            builder.setInsertionPointToStart(block);
            Value v =
                builder.create<quake::UnwrapOp>(arg.getLoc(), wireTy, arg);
            valMap.insert({arg, v});
            defMap[block].insert({arg, v});
          }
        }
      }

      // Loop over all operations in the block.
      for (Operation &oper : *block) {
        Operation *op = &oper;
        // For any operation that creates a value of quantum reference type,
        // replace it with a null wire (if it is an AllocaOp) or unwrap the
        // reference to get the wire.
        if (opResultOfType(op, qrefTy)) {
          if (!quantumValues)
            continue;
          // If this op defines a quantum reference, record it in the maps.
          if (auto alloc = dyn_cast<quake::AllocaOp>(op);
              alloc && memAnalysis.isMember(alloc)) {
            // If it is a known non-escaping alloca, then replace it with a null
            // wire and record it for removal.
            if (!defMap.count(block))
              defMap.insert({block, DefnMap{}});
            if (!defMap[block].count(alloc)) {
              OpBuilder builder(alloc);
              Value v =
                  builder.create<quake::NullWireOp>(alloc.getLoc(), wireTy);
              cleanUps.insert(op);
              valMap.insert({alloc, v});
              defMap[block].insert({alloc, v});
            }
          } else {
            OpBuilder builder(ctx);
            builder.setInsertionPointAfter(op);
            for (auto r : op->getResults()) {
              Value v =
                  builder.create<quake::UnwrapOp>(op->getLoc(), wireTy, r);
              valMap.insert({r, v});
              defMap[block].insert({r, v});
            }
          }
          continue;
        }

        // If this is a classical stack slot allocation (and we're processing
        // classical values), promote the allocation to an undefined value.
        if (auto alloc = dyn_cast<cudaq::cc::AllocaOp>(op);
            alloc && memAnalysis.isMember(alloc)) {
          if (classicalValues) {
            if (!defMap.count(block))
              defMap.insert({block, DefnMap{}});
            if (!defMap[block].count(alloc)) {
              OpBuilder builder(alloc);
              Value v = builder.create<cudaq::cc::UndefOp>(
                  alloc.getLoc(), alloc.getElementType());
              cleanUps.insert(op);
              valMap.insert({alloc, v});
              defMap[block].insert({alloc, v});
            }
          }
          continue;
        }

        // If this is a new value being created, add it to the map of values for
        // this block so it can be tracked and forwarded.
        if (auto nullWire = dyn_cast<quake::NullWireOp>(op)) {
          if (quantumValues)
            valMap.insert({nullWire, nullWire.getResult()});
          continue;
        }
        if (auto undef = dyn_cast<cudaq::cc::UndefOp>(op)) {
          if (classicalValues)
            valMap.insert({undef, undef.getResult()});
          continue;
        }

        auto handleUse = [&]<typename T>(T unwrap, Value memuse) {
          if (!memuse)
            return;
          if (valMap.count(memuse)) {
            if (!valMap[memuse]) {
              valMap[memuse] = unwrap;
            } else if (unwrap.getResult() != valMap[memuse]) {
              unwrap.replaceAllUsesWith(valMap[memuse]);
              cleanUps.insert(op);
            }
          } else if (block->isEntryBlock()) {
            if (isFunctionEntryBlock(block)) {
              // If this entry block is to a function, then we have a use
              // before a def and the IR is not is SSA form. We can't make
              // progress at this point so raise an error.
              oper.emitError("use before def in function");
              signalPassFailure();
              return;
            }
            auto newUnwrapVal = [&]() -> Value {
              if (promotedDefs.count(memuse))
                return promotedDefs[memuse];
              OpBuilder builder(parent);
              auto nu = cast<T>(builder.clone(*op));
              promotedDefs[memuse] = nu.getResult();
              return nu.getResult();
            }();
            if (parent->hasTrait<OpTrait::NoRegionArguments>()) {
              // In this case, parent does not accept region arguments so the
              // reference values must already be defined to dominate parent.
              unwrap.replaceAllUsesWith(newUnwrapVal);
            } else {
              // Otherwise, parent requires region arguments, so dominating
              // values must be added and threaded through the block
              // arguments.
              auto numOperands = parent->getNumOperands();
              parent->insertOperands(numOperands, ValueRange{newUnwrapVal});
              for (auto &reg : parent->getRegions()) {
                if (reg.empty())
                  continue;
                auto *entry = &reg.front();
                auto blockArg =
                    entry->addArgument(unwrap.getType(), unwrap.getLoc());
                defMap[entry][memuse] = blockArg;
                regionDefs[entry][memuse] = blockArg;
                if (unwrap->getParentRegion() == &reg)
                  unwrap.replaceAllUsesWith(blockArg);
                if (entry == block)
                  valMap[memuse] = blockArg;
              }
            }
            cleanUps.insert(unwrap);
          } else {
            // The def is not in running map and this is not an entry block.
            // We want to add the def to the arguments coming from our
            // predecessor blocks.
            auto &argVec = blockArgsMap[block];
            if (std::find(argVec.begin(), argVec.end(), memuse) ==
                argVec.end()) {
              // This one isn't already on the list of block arguments, so add
              // and record it as a new BlockArgument.
              argVec.push_back(memuse);
              auto newArg =
                  block->addArgument(unwrap.getType(), memuse.getLoc());
              valMap[memuse] = newArg;
              unwrap.replaceAllUsesWith(newArg);
              cleanUps.insert(op);
              // Iterate on all predecessors.
              for (auto *p : block->getPredecessors())
                worklist.push_back(p);
            }
          }
        };
        if (auto unwrap = dyn_cast<quake::UnwrapOp>(op)) {
          if (quantumValues)
            handleUse(unwrap, unwrap.getRefValue());
          continue;
        }
        if (auto load = dyn_cast<cudaq::cc::LoadOp>(op)) {
          if (classicalValues) {
            auto memuse = load.getPtrvalue();
            // Process only singleton classical scalars, no aggregates.
            if (auto *useOp = memuse.getDefiningOp())
              if (memAnalysis.isMember(useOp))
                handleUse(load, memuse);
          }
          continue;
        }

        auto handleDefinition = [&](Value val, Value memdef) {
          cleanUps.insert(op);
          valMap[memdef] = val;
          if (!isCallableOp && block->hasNoSuccessors() &&
              !isDescendantOf(parent, memdef)) {
            // Exit block of non-function. Record this wrap as a definition. It
            // may need to escape via the terminator.
            regionDefs[block][memdef] = val;
            cleanUps.insert(op);
          }
        };
        if (auto wrap = dyn_cast<quake::WrapOp>(op)) {
          if (quantumValues)
            handleDefinition(wrap.getWireValue(), wrap.getRefValue());
          continue;
        }
        if (auto store = dyn_cast<cudaq::cc::StoreOp>(op)) {
          if (classicalValues) {
            auto memuse = store.getPtrvalue();
            // Process only singleton classical scalars, no aggregates.
            if (auto *useOp = memuse.getDefiningOp())
              if (memAnalysis.isMember(useOp))
                handleDefinition(store.getValue(), store.getPtrvalue());
          }
          continue;
        }

        // If op uses a quantum reference, then halt forwarding the unwrap
        // use chain and leave a wrap dominating op.
        for (auto v : op->getOperands()) {
          if ((v.getType() == qrefTy) && valMap.count(v) && valMap[v]) {
            OpBuilder builder(op);
            builder.create<quake::WrapOp>(op->getLoc(), valMap[v], v);
            valMap[v] = Value{}; // null signals an unwrap is required.
          }
        }
      } // end of loop over ops in block

      if (block->hasNoSuccessors()) {
        // `block` is an exiting block. Nothing to do.
      } else {
        // `block` branches to other successor blocks in this region. Thread
        // live defs to the arguments of each successor block.
        auto branch = cast<BranchOpInterface>(block->getTerminator());
        for (auto iter : llvm::enumerate(block->getSuccessors())) {
          auto succBlock = iter.value();
          auto succNum = iter.index();
          auto argsToAdd = succBlock->getNumArguments() -
                           branch.getSuccessorOperands(succNum).size();
          if (argsToAdd > 0) {
            SmallVector<Value> newArguments;
            assert(blockArgsMap[succBlock].size() >= argsToAdd);
            // Only add the trailing ones that haven't been added yet.
            for (unsigned i = blockArgsMap[succBlock].size() - argsToAdd;
                 i < blockArgsMap[succBlock].size(); ++i) {
              auto defVal = blockArgsMap[succBlock][i];
              if (valMap.count(defVal)) {
                if (!valMap[defVal]) {
                  if (isa<quake::RefType>(defVal.getType())) {
                    // Wire was killed by an Op. Unwrap the reference again.
                    OpBuilder builder(ctx);
                    builder.setInsertionPoint(block->getTerminator());
                    auto unwrap = builder.create<quake::UnwrapOp>(
                        defVal.getLoc(), wireTy, defVal);
                    valMap[defVal] = unwrap;
                  } else {
                    OpBuilder builder(ctx);
                    builder.setInsertionPoint(block->getTerminator());
                    auto load = builder.create<cudaq::cc::LoadOp>(
                        defVal.getLoc(), defVal);
                    valMap[defVal] = load;
                  }
                }
                assert(valMap[defVal]);
                newArguments.push_back(valMap[defVal]);
              } else {
                auto &argVec = blockArgsMap[block];
                auto argIter = argVec.begin();
                assert(block->getNumArguments() >= argVec.size());
                unsigned j = block->getNumArguments() - argVec.size();
                while (argIter != argVec.end() && *argIter != defVal) {
                  argIter++;
                  j++;
                }
                if (argIter != argVec.end()) {
                  // Use the argument to this block as the branch argument.
                  newArguments.push_back(block->getArgument(j));
                } else {
                  // Add and record as a BlockArgument any value that is used in
                  // the successors but was neither used/defined in this block
                  // nor passed in as an argument.
                  argVec.push_back(defVal);
                  auto newArg = block->addArgument(unwrapType(defVal.getType()),
                                                   defVal.getLoc());
                  assert(newArg);
                  valMap[defVal] = newArg;
                  for (auto *p : block->getPredecessors())
                    worklist.push_back(p);
                  newArguments.push_back(newArg);
                }
              }
            }
            assert(newArguments.size() == argsToAdd);
            branch.getSuccessorOperands(succNum).append(newArguments);
          }
        }
      }

      appendPredecessorsToWorklist(worklist, block, blocksVisited);
    } // end of worklist loop

    // Thread the aggregate of all definitions through each exiting block.
    if (!isCallableOp) {
      // Determine all the unique definitions.
      SmallVector<Value> allDefs = getAllDefinitions(regionDefs);
      for (auto &defPair : regionDefs) {
        // For each exiting block, thread the values via the terminator.
        auto *terminator = defPair.first->getTerminator();
        SmallVector<Value> newArguments;
        for (unsigned i = 0; i < terminator->getNumOperands(); ++i)
          newArguments.push_back(terminator->getOperand(i));
        auto &defMap = defPair.second;
        for (auto v : allDefs)
          newArguments.push_back(defMap.count(v) ? defMap[v] : promotedDefs[v]);
        if (!newArguments.empty())
          terminator->setOperands(newArguments);
      }
      if (!allDefs.empty()) {
        // Replace parent with a copy.
        SmallVector<Type> resultTypes(parent->getResultTypes());
        for (auto d : allDefs)
          resultTypes.push_back(unwrapType(d.getType()));
        ConversionPatternRewriter builder(ctx);
        builder.setInsertionPoint(parent);
        SmallVector<Value> operands;
        for (auto opndVal : parent->getOperands())
          operands.push_back(opndVal);
        Operation *np =
            Operation::create(parent->getLoc(), parent->getName(), resultTypes,
                              operands, parent->getAttrs(),
                              parent->getSuccessors(), parent->getNumRegions());
        builder.insert(np);
        for (unsigned i = 0; i < parent->getNumRegions(); ++i)
          builder.inlineRegionBefore(parent->getRegion(i), np->getRegion(i),
                                     np->getRegion(i).begin());
        for (unsigned i = 0; i < parent->getNumResults(); ++i)
          parent->getResult(i).replaceAllUsesWith(np->getResult(i));
        builder.setInsertionPointAfter(np);
        for (auto iter : llvm::enumerate(allDefs)) {
          auto i = iter.index() + parent->getNumResults();
          if (np->getResult(i).getType() == wireTy)
            builder.create<quake::WrapOp>(np->getLoc(), np->getResult(i),
                                          iter.value());
          else
            builder.create<cudaq::cc::StoreOp>(np->getLoc(), np->getResult(i),
                                               iter.value());
        }
        cleanUps.insert(parent);
        parent = np;
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "After threading values:\n"
                            << *parent << "\n\n");
  }

  // Convert the function to "quantum load/store" (QLS) format.
  LogicalResult convertToQLS() {
    if (!quantumValues)
      return success();
    auto func = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<WRAPPER_QUANTUM_OPS, ResetOpPattern, DeallocOpPattern>(ctx);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<RAW_QUANTUM_OPS, quake::ResetOp,
                                 quake::DeallocOp>(
        [](Operation *op) { return !quake::hasNonVectorReference(op); });
    target.addLegalOp<quake::UnwrapOp, quake::WrapOp, quake::NullWireOp,
                      quake::SinkOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      emitError(func.getLoc(), "error converting to QLS form\n");
      signalPassFailure();
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "After converting to QLS:\n" << func << "\n\n");
    return success();
  }
};
} // namespace
