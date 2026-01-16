/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_REGTOMEM
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "regtomem"

using namespace mlir;

#define RAW(X) quake::X
#define RAW_MEASURE_OPS MEASURE_OPS(RAW)
#define RAW_GATE_OPS GATE_OPS(RAW)
#define RAW_QUANTUM_OPS QUANTUM_OPS(RAW)

static unsigned successorIndex(Operation *terminator, Block *successor) {
  for (auto iter : llvm::enumerate(terminator->getSuccessors()))
    if (iter.value() == successor)
      return iter.index();
  cudaq::emitFatalError(terminator->getLoc(), "successor block not found");
  return ~0u;
}

using BlockSet = SmallPtrSet<Block *, 4>;

namespace {
/// Register to memory analysis.
///
/// This is currently specific to quantum operations and wire values. It
/// computes all the equivalence classes to ascertain all mutually exclusive
/// wire values. If a particular wire's use comes from multiple wire
/// declarations (i.e., the specific op is shared by different parts of a
/// circuit), then transformation back to memory semantics is prevented.
///
/// Rationale: Adding quantum references must be handled conservatively. For
/// instance, should CSE perform block merging, it would naively be
/// straightforward to introduce new quantum references that serve as copies of
/// qubit wire values. However, introduction of such phantom qubit copies is not
/// allowed as it is not possible to copy a qubit.
struct RegToMemAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RegToMemAnalysis)

  RegToMemAnalysis(func::FuncOp f) { performAnalysis(f); }

  bool failed() const { return analysisFailed; }

  // This is the cardinality of the number of null wires in the function.
  unsigned getCardinality() const { return cardinality; }

  std::optional<unsigned> idFromValue(Value v) const {
    auto iter = eqClasses.findValue(toOpaque(v));
    if (iter == eqClasses.end())
      return std::nullopt;
    return setIds.find(eqClasses.getLeaderValue(toOpaque(v)))->second;
  }

  ArrayRef<Operation *> getWires() const { return theWires; }

  ArrayRef<quake::UnwrapOp> getUnwrapReferences() const { return unwrapRefs; }

private:
  void *toOpaque(Value v) const { return v.getAsOpaquePointer(); }

  void insertBlockArgumentToEqClass(Value v) {
    if (auto arg = dyn_cast_or_null<BlockArgument>(v)) {
      auto *block = arg.getOwner();
      auto argNum = arg.getArgNumber();
      for (auto *pred : block->getPredecessors()) {
        auto *term = pred->getTerminator();
        auto i = successorIndex(term, block);
        Value u = cast<BranchOpInterface>(term).getSuccessorOperands(i)[argNum];
        if (eqClasses.findValue(toOpaque(u)) == eqClasses.end())
          insertToEqClass(u, v);
        else
          eqClasses.unionSets(toOpaque(v), toOpaque(u));
      }
    }
  }

  void insertToEqClass(Value v) {
    if (quake::isLinearValueForm(v))
      eqClasses.insert(toOpaque(v));
    insertBlockArgumentToEqClass(v);
  }

  /// Is this a quake.unwrap that is in our definitions set?
  bool isUnwrapRef(Value v) {
    if (auto unwrap = v.getDefiningOp<quake::UnwrapOp>())
      return std::find(unwrapRefs.begin(), unwrapRefs.end(), unwrap) !=
             unwrapRefs.end();
    return false;
  }

  void insertToEqClass(Value v, Value u) {
    LLVM_DEBUG(llvm::dbgs() << "add eqv of " << v << " and " << u << "\n");
    if (quake::isLinearValueForm(v) || isUnwrapRef(v))
      eqClasses.unionSets(toOpaque(v), toOpaque(u));
    insertBlockArgumentToEqClass(v);
  }

  // Multiple borrows of the same wire count as only one unique wire. Ensure
  // that we count them correctly.
  void collectBorrowWires(func::FuncOp func) {
    DenseMap<StringRef, DenseMap<std::int32_t, Value>> uniqBorrows;
    func.walk([&](quake::BorrowWireOp borrow) {
      LLVM_DEBUG(llvm::dbgs() << "adding borrow : " << borrow << '\n');
      theWires.push_back(borrow.getOperation());
      auto iter = uniqBorrows.find(borrow.getSetName());
      if (iter == uniqBorrows.end()) {
        uniqBorrows[borrow.getSetName()][borrow.getIdentity()] = borrow;
        eqClasses.insert(toOpaque(borrow));
        ++cardinality;
        return;
      }
      auto iter2 = iter->second.find(borrow.getIdentity());
      if (iter2 == iter->second.end()) {
        uniqBorrows[borrow.getSetName()][borrow.getIdentity()] = borrow;
        eqClasses.insert(toOpaque(borrow));
        ++cardinality;
        return;
      }
      insertToEqClass(iter2->second, borrow);
    });
  }

  // Collect all linear type values from a generic iterable range into a
  // temporary vector.
  template <typename A>
  SmallVector<Value> collectLinearValues(A &&vals) {
    SmallVector<Value> result;
    for (Value v : vals)
      if (quake::isLinearType(v.getType()))
        result.push_back(v);
    return result;
  }
  void performAnalysis(func::FuncOp func) {
    collectBorrowWires(func);
    func.walk([&](Operation *op) {
      if (auto nwire = dyn_cast<quake::NullWireOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "adding |0> : " << nwire << '\n');
        theWires.push_back(nwire.getOperation());
        eqClasses.insert(toOpaque(nwire));
        ++cardinality;
      } else if (auto unwrap = dyn_cast<quake::UnwrapOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "adding unwrap: " << unwrap << '\n');
        unwrapRefs.push_back(unwrap);
        eqClasses.insert(toOpaque(unwrap));
        ++cardinality;
      }
    });
    func.walk([&](Operation *op) {
      if (isa<RAW_MEASURE_OPS>(op)) {
        auto wireOpnds = collectLinearValues(op->getOperands());
        for (auto [t, r] : llvm::zip(wireOpnds, op->getResults().drop_front()))
          insertToEqClass(t, r);
      } else if (isa<RAW_GATE_OPS>(op)) {
        auto gate = cast<quake::OperatorInterface>(op);
        // Check that the IR is not in the pruned control form.
        if (llvm::any_of(gate.getControls(), [](Value v) {
              return isa<quake::ControlType>(v.getType());
            })) {
          op->emitError("must use linear-ctrl-form before regtomem");
          analysisFailed = true;
          return;
        }
        auto wireCtrls = collectLinearValues(gate.getControls());
        for (auto [t, r] : llvm::zip(wireCtrls, op->getResults()))
          insertToEqClass(t, r);
        auto wireTargs = collectLinearValues(gate.getTargets());
        for (auto [t, r] : llvm::zip(
                 wireTargs, op->getResults().drop_front(wireCtrls.size())))
          insertToEqClass(t, r);
      } else if (auto reset = dyn_cast<quake::ResetOp>(op)) {
        auto wireTargs =
            collectLinearValues(ArrayRef<Value>{reset.getTargets()});
        for (auto [t, r] : llvm::zip(wireTargs, reset.getResults()))
          insertToEqClass(t, r);
      } else if (auto sink = dyn_cast<quake::SinkOp>(op)) {
        insertToEqClass(sink.getTarget());
      } else if (auto ret = dyn_cast<quake::ReturnWireOp>(op)) {
        insertToEqClass(ret.getTarget());
      } else if (auto ccif = dyn_cast<cudaq::cc::IfOp>(op)) {
        if (!ccif.getLinearArgs().empty()) {
          if (!ccif.hasThen() || !ccif.hasElse()) {
            analysisFailed = true;
            return;
          }
          for (auto [v, u] :
               llvm::zip(ccif.getThenEntryArguments(), ccif.getLinearArgs()))
            insertToEqClass(v, u);
          for (auto [v, u] :
               llvm::zip(ccif.getElseEntryArguments(), ccif.getLinearArgs()))
            insertToEqClass(v, u);
        }
      } else if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(op)) {
        auto *parent = cont->getParentOp();
        if (auto ccif = dyn_cast<cudaq::cc::IfOp>(parent)) {
          for (auto iter : llvm::enumerate(cont.getOperands()))
            if (quake::isLinearType(iter.value().getType()))
              insertToEqClass(ccif.getResult(iter.index()), iter.value());
        } else if (isa<cudaq::cc::ScopeOp>(parent)) {
          if (llvm::any_of(cont.getOperands(), [](Value v) {
                return quake::isQuantumValueType(v.getType());
              })) {
            analysisFailed = true;
            return;
          }
        } else {
          // TODO: handle cc::LoopOp.
          analysisFailed = true;
          return;
        }
      } else if (auto branch = dyn_cast<BranchOpInterface>(op)) {
        const unsigned numSuccs = branch->getNumSuccessors();
        for (unsigned i = 0; i < numSuccs; ++i) {
          auto succOperands = branch.getSuccessorOperands(i);
          for (auto [so, fo] :
               llvm::zip(branch->getSuccessor(i)->getArguments(),
                         succOperands.getForwardedOperands()))
            if (quake::isLinearType(so.getType()))
              insertToEqClass(so, fo);
        }
      }
    });
    LLVM_DEBUG(llvm::dbgs() << "The cardinality " << cardinality
                            << ". The number of equivalence classes "
                            << eqClasses.getNumClasses() << ".\n");
    if (cardinality != eqClasses.getNumClasses()) {
      analysisFailed = true;
      return;
    }
    unsigned id = 0;
    for (auto i = eqClasses.begin(), end = eqClasses.end(); i != end; ++i)
      if (i->isLeader()) {
        void *leader = const_cast<void *>(*eqClasses.findLeader(i));
        setIds.insert(std::make_pair(leader, id++));
      }
  }

  // For debugging purposes.
  void dump() const {
    for (auto i = eqClasses.begin(); i != eqClasses.end(); ++i) {
      if (!i->isLeader())
        continue;
      llvm::errs() << "Set {\n";
      for (auto e = eqClasses.member_begin(i); e != eqClasses.member_end(); ++e)
        llvm::errs() << "  " << Value::getFromOpaquePointer(*e) << '\n';
      llvm::errs() << "}\n";
    }
  }

  SmallVector<Operation *> theWires;
  SmallVector<quake::UnwrapOp> unwrapRefs;
  llvm::EquivalenceClasses<void *> eqClasses;
  DenseMap<void *, unsigned> setIds;
  unsigned cardinality = 0;
  bool analysisFailed = false;
};

template <typename OP>
class CollapseWrappers : public OpRewritePattern<OP> {
public:
  using Base = OpRewritePattern<OP>;
  explicit CollapseWrappers(MLIRContext *ctx, RegToMemAnalysis &analysis,
                            ArrayRef<Value> allocas)
      : Base(ctx), analysis(analysis), allocas(allocas) {}

  LogicalResult matchAndRewrite(OP op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto findLookupValue = [&](Value v) -> Value {
      if (auto id = analysis.idFromValue(v))
        return allocas[*id];
      if (auto u = v.template getDefiningOp<quake::UnwrapOp>())
        return u.getRefValue();
      return v;
    };
    auto collect = [&](auto values) {
      SmallVector<Value> args;
      for (auto v : values)
        args.push_back(findLookupValue(v));
      return args;
    };
    auto eraseWrapUsers = [&](auto op) {
      for (auto *usr : op->getUsers())
        if (isa<quake::WrapOp>(usr))
          rewriter.eraseOp(usr);
    };

    if constexpr (quake::isMeasure<OP>) {
      auto args = collect(op.getOperands());
      auto nameAttr = op.getRegisterNameAttr();
      eraseWrapUsers(op);
      auto newOp = rewriter.create<OP>(
          loc, ArrayRef<Type>{op.getMeasOut().getType()}, args, nameAttr);
      op.getResult(0).replaceAllUsesWith(newOp.getResult(0));
      rewriter.eraseOp(op);
    } else if constexpr (std::is_same_v<OP, quake::ResetOp>) {
      // Reset is a special case.
      auto targ = findLookupValue(op.getTargets());
      eraseWrapUsers(op);
      rewriter.create<quake::ResetOp>(loc, TypeRange{}, targ);
      rewriter.eraseOp(op);
    } else if constexpr (std::is_same_v<OP, quake::SinkOp>) {
      auto targ = findLookupValue(op.getTarget());
      rewriter.replaceOpWithNewOp<quake::DeallocOp>(op, targ);
    } else if constexpr (std::is_same_v<OP, quake::ReturnWireOp>) {
      rewriter.eraseOp(op);
    } else {
      auto ctrls = collect(op.getControls());
      auto targs = collect(op.getTargets());
      eraseWrapUsers(op);
      rewriter.create<OP>(loc, op.getIsAdj(), op.getParameters(), ctrls, targs,
                          op.getNegatedQubitControlsAttr());
      rewriter.eraseOp(op);
    }
    return success();
  }

  RegToMemAnalysis &analysis;
  ArrayRef<Value> allocas;
};

struct EraseWiresBranch : public OpRewritePattern<cf::BranchOp> {
  explicit EraseWiresBranch(MLIRContext *ctx, BlockSet &blocks)
      : OpRewritePattern(ctx), blocks(blocks) {}

  LogicalResult matchAndRewrite(cf::BranchOp branch,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newOperands;
    for (auto v : branch.getDestOperands()) {
      if (quake::isLinearType(v.getType()))
        blocks.insert(branch.getDest());
      else
        newOperands.push_back(v);
    }
    rewriter.replaceOpWithNewOp<cf::BranchOp>(branch, newOperands,
                                              branch.getDest());
    return success();
  }

  BlockSet &blocks;
};

struct EraseWiresCondBranch : public OpRewritePattern<cf::CondBranchOp> {
  explicit EraseWiresCondBranch(MLIRContext *ctx, BlockSet &blocks)
      : OpRewritePattern(ctx), blocks(blocks) {}

  LogicalResult matchAndRewrite(cf::CondBranchOp branch,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newTrueOperands;
    for (auto v : branch.getTrueDestOperands()) {
      if (quake::isLinearType(v.getType()))
        blocks.insert(branch.getTrueDest());
      else
        newTrueOperands.push_back(v);
    }
    SmallVector<Value> newFalseOperands;
    for (auto v : branch.getFalseDestOperands()) {
      if (quake::isLinearType(v.getType()))
        blocks.insert(branch.getFalseDest());
      else
        newFalseOperands.push_back(v);
    }
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        branch, branch.getCondition(), newTrueOperands, newFalseOperands,
        branch.getTrueDest(), branch.getFalseDest());
    return success();
  }
  BlockSet &blocks;
};

struct EraseWiresIf : public OpRewritePattern<cudaq::cc::IfOp> {
  explicit EraseWiresIf(MLIRContext *ctx, RegToMemAnalysis &analysis,
                        ArrayRef<Value> allocas)
      : OpRewritePattern(ctx), analysis(analysis), allocas(allocas) {}

  // Rewriting the cc.if operation is done in a single step here. It can
  // probably be decomposed into smaller steps. We eliminate the original
  // cc.if's wire arguments, entry block arguments, prune the cc.continue's
  // return list, and replace any users of the wire outputs with quake.unwrap
  // operations.
  LogicalResult matchAndRewrite(cudaq::cc::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto wireTy = quake::WireType::get(ctx);

    // Create a new if operation, pruning the result type and discarding the
    // operands of the original if operation.
    SmallVector<Type> newIfTy;
    for (auto ty : ifOp.getResultTypes())
      if (!quake::isLinearType(ty))
        newIfTy.push_back(ty);
    auto origThenArgs = ifOp.getThenRegion().front().getArguments();
    auto origElseArgs = ifOp.getElseRegion().front().getArguments();
    auto newIf = rewriter.create<cudaq::cc::IfOp>(
        ifOp.getLoc(), newIfTy, ifOp.getCondition(),
        [&](OpBuilder &, Location, Region &region) {
          rewriter.inlineRegionBefore(ifOp.getThenRegion(), region,
                                      region.end());
        },
        [&](OpBuilder &, Location, Region &region) {
          rewriter.inlineRegionBefore(ifOp.getElseRegion(), region,
                                      region.end());
        });

    // Erase entry block arguments and prune the cc.continue operations.
    auto replaceArgsContinues = [&](Region &region,
                                    MutableArrayRef<BlockArgument> origArgs) {
      auto &entry = region.front();
      const unsigned count = entry.getNumArguments();
      {
        OpBuilder builder(ctx);
        builder.setInsertionPointToStart(&entry);
        for (auto [arg, from] : llvm::zip(entry.getArguments(), origArgs)) {
          auto id = analysis.idFromValue(from);
          assert(id);
          auto unwrap = builder.create<quake::UnwrapOp>(ifOp.getLoc(), wireTy,
                                                        allocas[*id]);
          arg.replaceAllUsesWith(unwrap);
        }
      }
      entry.eraseArguments(0, count);
      for (auto &block : region)
        for (auto &op : block)
          if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(op)) {
            SmallVector<Value> newOpnds;
            OpBuilder builder(cont);
            for (auto v : cont.getOperands())
              if (!quake::isLinearType(v.getType()))
                newOpnds.push_back(v);
            builder.create<cudaq::cc::ContinueOp>(cont.getLoc(), newOpnds);
            rewriter.eraseOp(cont);
          }
    };
    replaceArgsContinues(newIf.getThenRegion(), origThenArgs);
    replaceArgsContinues(newIf.getElseRegion(), origElseArgs);

    // Replace any original uses with uses of a mix of the new if op's values
    // (if not type wire) or quake.unwrap operations (only if type wire).
    SmallVector<Value> unwraps;
    unsigned i = 0;
    for (auto v : ifOp.getResults()) {
      if (quake::isLinearType(v.getType())) {
        auto id = analysis.idFromValue(v);
        assert(id);
        auto unwrap = rewriter.create<quake::UnwrapOp>(ifOp.getLoc(), wireTy,
                                                       allocas[*id]);
        unwraps.push_back(unwrap);
      } else {
        unwraps.push_back(newIf.getResult(i++));
      }
    }
    rewriter.replaceOp(ifOp, unwraps);
    return success();
  }

  RegToMemAnalysis &analysis;
  ArrayRef<Value> allocas;
};

#define NOWRAP(OP) CollapseWrappers<quake::OP>
#define NOWRAP_QUANTUM_OPS QUANTUM_OPS(NOWRAP)

class RegToMemPass : public cudaq::opt::impl::RegToMemBase<RegToMemPass> {
public:
  using RegToMemBase::RegToMemBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto *ctx = &getContext();

    // 1) Run the analysis to find all potential variables and create their
    // equivalence classes.
    RegToMemAnalysis analysis(func);
    if (analysis.failed()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "regtomem dataflow analysis for \"" << func.getName()
                 << "\" failed.\nAll wire uses must remain disjoint from the "
                    "source null/borrow to the sink/return operation.\n");
      return;
    }

    // 2) Replace each quake.null_wire with a quake.alloca.
    SmallVector<Value> allocas(analysis.getCardinality());
    SmallVector<Value> borrowAllocas;
    for (auto *nwire : llvm::reverse(analysis.getWires())) {
      OpBuilder builder(ctx);
      const bool fromWire = isa<quake::BorrowWireOp>(nwire);
      if (fromWire)
        builder.setInsertionPointToStart(&func.getBody().front());
      else
        builder.setInsertionPoint(nwire);
      auto qrefTy = quake::RefType::get(ctx);
      Value a =
          builder.create<quake::AllocaOp>(nwire->getLoc(), qrefTy, Value{});
      if (fromWire)
        borrowAllocas.push_back(a);
      if (auto opt = analysis.idFromValue(nwire->getResult(0))) {
        allocas[*opt] = a;
      } else {
        nwire->emitOpError("analysis failed: wire not seen\n");
        signalPassFailure();
        return;
      }
    }
    for (auto unwrap : analysis.getUnwrapReferences()) {
      if (auto opt = analysis.idFromValue(unwrap)) {
        allocas[*opt] = unwrap.getRefValue();
      } else {
        unwrap.emitOpError("analysis failed (unwrap)\n");
        signalPassFailure();
        return;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "After placing allocations:\n"
                            << func << "\n\n");

    // 3) Replace each gate pattern (either wrapped or value-ssa form) with a
    // gate in memory-ssa form.
    auto hasNoWires = [](Operation *op) {
      return op->getOperands().empty() ||
             !llvm::any_of(op->getOperands(), [](Value v) {
               return v && quake::isLinearType(v.getType());
             });
    };
    BlockSet fixupBlocks;
    RewritePatternSet patterns(ctx);
    patterns.insert<NOWRAP_QUANTUM_OPS, CollapseWrappers<quake::ResetOp>,
                    CollapseWrappers<quake::ReturnWireOp>,
                    CollapseWrappers<quake::SinkOp>, EraseWiresIf>(
        ctx, analysis, allocas);
    patterns.insert<EraseWiresBranch, EraseWiresCondBranch>(ctx, fixupBlocks);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<RAW_QUANTUM_OPS, quake::ResetOp, cf::BranchOp,
                                 cf::CondBranchOp, cudaq::cc::IfOp>(
        [&](Operation *op) { return hasNoWires(op); });
    target.addIllegalOp<quake::SinkOp, quake::ReturnWireOp>();
    target.addLegalOp<quake::UnwrapOp, quake::DeallocOp>();
    target.addLegalDialect<cudaq::cc::CCDialect>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      func.emitError("error converting to memory form\n");
      signalPassFailure();
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "After converting to memory SSA form:\n"
                            << func << "\n\n");

    // 4) Cleanup all the block arguments, NullWireOp, or UnwrapOp.
    cleanupBlocks(fixupBlocks);
    func.walk([&](Operation *op) -> WalkResult {
      if (isa<quake::NullWireOp, quake::BorrowWireOp, quake::UnwrapOp>(op) &&
          op->getUses().empty()) {
        op->erase();
        return WalkResult::skip();
      }
      if (isa<func::ReturnOp>(op) && !borrowAllocas.empty()) {
        OpBuilder builder(op);
        for (auto v : borrowAllocas)
          builder.create<quake::DeallocOp>(func.getLoc(), v);
      }
      return WalkResult::advance();
    });
    LLVM_DEBUG(llvm::dbgs() << "After cleanup:\n" << func << "\n\n");
  }

  void cleanupBlocks(const BlockSet &blocks) {
    for (auto *b : blocks) {
      unsigned i = 0;
      for (auto arg : b->getArguments()) {
        if (quake::isLinearType(arg.getType()))
          b->eraseArgument(i);
        else
          ++i;
      }
    }
  }
};
} // namespace
