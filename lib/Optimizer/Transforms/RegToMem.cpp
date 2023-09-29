/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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

  ArrayRef<quake::NullWireOp> getNullWires() const { return nullWires; }

  ArrayRef<quake::UnwrapOp> getUnwrapReferences() const { return unwrapRefs; }

  SmallVector<BlockArgument> getBlockArguments() const {
    SmallVector<BlockArgument> result;
    for (auto i = eqClasses.begin(); i != eqClasses.end(); ++i) {
      if (!i->isLeader())
        continue;
      for (auto e = eqClasses.member_begin(i); e != eqClasses.member_end();
           ++e) {
        auto v = Value::getFromOpaquePointer(*e);
        if (auto ba = v.dyn_cast_or_null<BlockArgument>())
          result.push_back(ba);
      }
    }
    return result;
  }

private:
  void *toOpaque(Value v) const { return v.getAsOpaquePointer(); }

  void insertBlockArgumentToEqClass(Value v) {
    if (auto arg = v.dyn_cast_or_null<BlockArgument>()) {
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
    if (quake::isValueSSAForm(v))
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
    if (quake::isValueSSAForm(v) || isUnwrapRef(v))
      eqClasses.unionSets(toOpaque(v), toOpaque(u));
    insertBlockArgumentToEqClass(v);
  }

  void performAnalysis(func::FuncOp func) {
    func.walk([&](Operation *op) {
      if (auto nwire = dyn_cast<quake::NullWireOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "adding |0> : " << nwire << '\n');
        nullWires.push_back(nwire);
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
        for (auto v : op->getOperands())
          insertToEqClass(v);
      } else if (isa<RAW_GATE_OPS>(op)) {
        auto gate = cast<quake::OperatorInterface>(op);
        for (auto c : gate.getControls())
          insertToEqClass(c);
        for (auto [t, r] : llvm::zip(gate.getTargets(), op->getResults()))
          insertToEqClass(t, r);
      } else if (auto reset = dyn_cast<quake::ResetOp>(op)) {
        insertToEqClass(reset.getTargets(), reset.getResult(0));
      } else if (auto sink = dyn_cast<quake::SinkOp>(op)) {
        insertToEqClass(sink.getTarget());
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

  SmallVector<quake::NullWireOp> nullWires;
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
      rewriter.replaceOpWithNewOp<OP>(
          op, ArrayRef<Type>{op.getBits().getType()}, args, nameAttr);
    } else if constexpr (std::is_same_v<OP, quake::ResetOp>) {
      // Reset is a special case.
      auto targ = findLookupValue(op.getTargets());
      eraseWrapUsers(op);
      rewriter.create<quake::ResetOp>(loc, TypeRange{}, targ);
      rewriter.eraseOp(op);
    } else if constexpr (std::is_same_v<OP, quake::SinkOp>) {
      auto targ = findLookupValue(op.getTarget());
      eraseWrapUsers(op);
      rewriter.replaceOpWithNewOp<quake::DeallocOp>(op, targ);
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
      LLVM_DEBUG(llvm::dbgs() << func << "\n");
      func.emitOpError("register analysis failed\n");
      signalPassFailure();
      return;
    }

    // 2) Replace each quake.null_wire with a quake.alloca.
    SmallVector<Value> allocas(analysis.getCardinality());
    for (auto nwire : analysis.getNullWires()) {
      OpBuilder builder(nwire);
      auto qrefTy = quake::RefType::get(ctx);
      Value a =
          builder.create<quake::AllocaOp>(nwire.getLoc(), qrefTy, Value{});
      if (auto opt = analysis.idFromValue(nwire)) {
        allocas[*opt] = a;
      } else {
        nwire.emitOpError("analysis failed (null_wire)\n");
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

    // 3) Erase any block arguments used to thread wire values.
    for (auto ba : analysis.getBlockArguments())
      eraseBlockArgumentFromBranch(ba);

    // 4) Replace each gate pattern (either wrapped or value-ssa form) with a
    // gate in memory-ssa form.
    RewritePatternSet patterns(ctx);
    patterns.insert<NOWRAP_QUANTUM_OPS, CollapseWrappers<quake::ResetOp>,
                    CollapseWrappers<quake::SinkOp>>(ctx, analysis, allocas);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<RAW_QUANTUM_OPS, quake::ResetOp>(
        [](Operation *op) { return quake::isAllReferences(op); });
    target.addIllegalOp<quake::SinkOp>();
    target.addLegalOp<quake::DeallocOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      emitError(func.getLoc(), "error converting to memory form\n");
      signalPassFailure();
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "After converting to memory SSA form:\n"
                            << func << "\n\n");

    // 4) Cleanup any unused block arguments, NullWireOp, or UnwrapOp.
    for (auto ba : analysis.getBlockArguments())
      eraseBlockArgumentFromBlock(ba);
    func.walk([&](Operation *op) {
      if (isa<quake::NullWireOp, quake::UnwrapOp>(op) && op->getUses().empty())
        op->erase();
    });
    LLVM_DEBUG(llvm::dbgs() << "After cleanup:\n" << func << "\n\n");
  }

  void eraseBlockArgumentFromBranch(BlockArgument ba) {
    auto *block = ba.getOwner();
    auto argNum = ba.getArgNumber();
    for (auto *pred : block->getPredecessors()) {
      auto *term = pred->getTerminator();
      auto i = successorIndex(term, block);
      auto succOperands = cast<BranchOpInterface>(term).getSuccessorOperands(i);
      succOperands.erase(argNum);
    }
  }

  void eraseBlockArgumentFromBlock(BlockArgument ba) {
    if (auto *block = ba.getOwner())
      if (!block->isEntryBlock())
        block->eraseArgument(ba.getArgNumber());
  }
};
} // namespace
