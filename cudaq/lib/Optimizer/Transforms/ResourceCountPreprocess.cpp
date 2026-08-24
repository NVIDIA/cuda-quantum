/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_RESOURCECOUNTPREPROCESS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "resource-count-preprocess"

using namespace mlir;

struct ResourceCountPreprocessPass
    : public cudaq::opt::impl::ResourceCountPreprocessBase<
          ResourceCountPreprocessPass> {
  using ResourceCountPreprocessBase::ResourceCountPreprocessBase;
  SetVector<Operation *> to_erase;
  DenseMap<Value, std::size_t> qubitIndexMap;
  std::size_t nextQubitIndex = 0;

  /// Assign a base qubit index for a qvector Value. For sized veqs, advances
  /// nextQubitIndex by the veq size so each qubit gets a unique index. For
  /// unsized veqs, the base index is shared (nextQubitIndex is not advanced)
  /// since individual qubits cannot be resolved.
  std::size_t getVeqBase(Value veq) {
    auto it = qubitIndexMap.find(veq);
    if (it != qubitIndexMap.end())
      return it->second;
    auto base = nextQubitIndex;
    if (auto size = cudaq::quake::getVeqSize(veq))
      nextQubitIndex += *size;
    qubitIndexMap[veq] = base;
    return base;
  }

  /// A wire carried into a loop body arrives as a block argument. The wire at
  /// index `i` of the body is the wire at index `i` of the loop's initial
  /// arguments, but only if it stays at that index on every edge of the loop.
  /// A body that hands its wires back permuted would otherwise attribute gates
  /// to the wrong qubit. Returns a null Value when that cannot be established.
  Value getLoopInitOperand(BlockArgument arg) {
    auto loop =
        dyn_cast_or_null<cudaq::cc::LoopOp>(arg.getOwner()->getParentOp());
    if (!loop || arg.getOwner() != loop.getDoEntryBlock())
      return {};
    auto i = arg.getArgNumber();
    if (i >= loop.getInitialArgs().size() ||
        i >= loop.getWhileArguments().size())
      return {};

    // The while region forwards its own arguments to the body unpermuted.
    auto cond =
        dyn_cast<cudaq::cc::ConditionOp>(loop.getWhileBlock()->getTerminator());
    if (!cond || i >= cond.getResults().size() ||
        cond.getResults()[i] != loop.getWhileArguments()[i])
      return {};

    // The body and the step region hand the wire back at the same index.
    for (Region *region : {&loop.getBodyRegion(), &loop.getStepRegion()}) {
      if (region->empty())
        continue;
      if (!region->hasOneBlock())
        return {};
      Block &block = region->front();
      auto *term = block.getTerminator();
      if (i >= block.getNumArguments() || i >= term->getNumOperands())
        return {};
      // Stop at block arguments here; following them would come straight back
      // to this loop.
      if (getWireOrigin(term->getOperand(i), /*throughLoops=*/false) !=
          block.getArgument(i))
        return {};
    }
    return loop.getInitialArgs()[i];
  }

  /// Walk a wire back to the value that introduced it. Quantum operations in
  /// value (linear) form thread wires through, so a wire result denotes the
  /// same qubit as the incoming wire in the matching position. `unwrap` bridges
  /// from reference form, so continue the search on the reference it wraps.
  /// `to_ctrl` and `from_ctrl` convert between wire and control form without
  /// changing the qubit, so continue on the value they convert. A wire entering
  /// an invariant loop body continues at the loop's matching initial argument.
  /// Returns the original value unchanged for anything else, including
  /// reference form values.
  Value getWireOrigin(Value v, bool throughLoops = true) {
    while (true) {
      auto *def = v.getDefiningOp();
      if (!def) {
        auto arg = cast<BlockArgument>(v);
        Value init;
        if (throughLoops)
          init = getLoopInitOperand(arg);
        if (!init)
          return v;
        v = init;
        continue;
      }
      if (auto unwrap = dyn_cast<cudaq::quake::UnwrapOp>(def)) {
        v = unwrap.getRefValue();
        continue;
      }
      if (auto toCtrl = dyn_cast<cudaq::quake::ToControlOp>(def)) {
        v = toCtrl.getQubit();
        continue;
      }
      if (auto fromCtrl = dyn_cast<cudaq::quake::FromControlOp>(def)) {
        v = fromCtrl.getCtrlbit();
        continue;
      }
      auto flow = cudaq::quake::detail::getThreadedWireFlow(def);
      if (!flow)
        return v;
      auto iter = llvm::find(flow->results, v);
      if (iter == flow->results.end())
        return v;
      v = flow->inputs[std::distance(flow->results.begin(), iter)];
    }
  }

  /// Resolve a quake value to a globally unique qubit index.
  std::optional<std::size_t> resolveQubitIndex(Value value) {
    Value v = getWireOrigin(value);
    // extract_ref from a qvector: base offset + local index. The index is
    // either an attribute or an operand that may be a constant.
    if (auto extractRef = v.getDefiningOp<cudaq::quake::ExtractRefOp>()) {
      if (extractRef.hasConstantIndex())
        return getVeqBase(extractRef.getVeq()) + extractRef.getConstantIndex();
      if (auto index =
              cudaq::opt::factory::maybeValueOfIntConstant(extractRef.getIndex()))
        return getVeqBase(extractRef.getVeq()) + *index;
    }
    // Wire semantics: concrete physical index from routing.
    if (auto borrow = v.getDefiningOp<cudaq::quake::BorrowWireOp>())
      return static_cast<std::size_t>(borrow.getIdentity());
    // Single-qubit alloca or a virtual null wire: assign a unique index by
    // declaration order.
    if (v.getDefiningOp<cudaq::quake::NullWireOp>() ||
        (v.getDefiningOp<cudaq::quake::AllocaOp>() &&
         isa<cudaq::quake::RefType>(v.getType()))) {
      auto it = qubitIndexMap.find(v);
      if (it != qubitIndexMap.end())
        return it->second;
      auto idx = nextQubitIndex++;
      qubitIndexMap[v] = idx;
      return idx;
    }
    return std::nullopt;
  }

  /// Resolve the qubit indices of `operands` into `out`. Returns false when
  /// any index does not resolve.
  bool resolveOperands(ValueRange operands, std::vector<std::size_t> &out) {
    bool allResolved = true;
    for (auto val : operands) {
      // A veq control may be gathered by `concat` (Python list syntax:
      // x.ctrl([q0, q1], q2)); resolve the qubits it holds.
      SmallVector<Value> qubits;
      if (auto concat = val.getDefiningOp<cudaq::quake::ConcatOp>())
        llvm::append_range(qubits, concat.getTargets());
      else
        qubits.push_back(val);
      for (auto qubit : qubits) {
        if (auto idx = resolveQubitIndex(qubit))
          out.push_back(*idx);
        else
          allResolved = false;
      }
    }
    return allResolved;
  }

  /// In value (linear) form each wire result of an operation is the outgoing
  /// value of the corresponding incoming wire. Forward the incoming wires to
  /// the uses of the results so the wire chain stays intact once the
  /// pre-counted operation is erased. Reference form has no wire results and
  /// is a no-op here.
  static void forwardWires(ValueRange incoming, ValueRange wires) {
    for (auto [in, out] : llvm::zip(incoming, wires))
      out.replaceAllUsesWith(in);
  }

  bool preCount(Operation *op, size_t to_add) {
    if (!isQuakeOperation(op))
      return false;

    if (auto measurement = dyn_cast<cudaq::quake::MeasurementInterface>(op);
        isa<cudaq::quake::MxOp, cudaq::quake::MyOp>(op)) {
      // An unread measurement cannot affect resource-count control flow. Count
      // it from Quake IR before code generation lowers its axis to an execution
      // manager Z measurement, then remove it with the other pre-counted ops.
      // The wires it threads are forwarded to its users, as for any other op.
      auto measWires = measurement.getWires();
      auto flow = cudaq::quake::detail::getThreadedWireFlow(op);
      if (!flow)
        return false;
      for (Value result : op->getResults())
        if (!llvm::is_contained(measWires, result) && !result.use_empty())
          return false;

      // A qubit index that does not resolve would be counted against the
      // wrong qubit. Leave the operation in the IR to be counted at run time.
      std::vector<std::size_t> targetIndices;
      if (!resolveOperands(measurement.getTargets(), targetIndices))
        return false;

      auto name = op->getName().stripDialect();
      if (dumpPreprocessed)
        llvm::outs() << "Preprocessing " << name << "(0) for " << to_add
                     << " counts\n";
      countGate(name.str(), {}, targetIndices, to_add);
      forwardWires(flow->inputs, measWires);
      to_erase.insert(op);
      return true;
    }

    auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);

    if (!opi)
      return false;

    auto wires = opi.getWires();
    auto flow = cudaq::quake::detail::getThreadedWireFlow(op);
    if (!flow)
      return false;

    auto name = op->getName().stripDialect();

    // A qubit index that does not resolve would be counted with the wrong
    // control count, arity and depth. Leave the operation in the IR to be
    // counted at run time.
    std::vector<std::size_t> controlIndices, targetIndices;
    bool resolved = resolveOperands(opi.getControls(), controlIndices);
    resolved &= resolveOperands(opi.getTargets(), targetIndices);
    if (!resolved)
      return false;

    if (dumpPreprocessed)
      llvm::outs() << "Preprocessing " << name << "("
                   << opi.getControls().size() << ")"
                   << " for " << to_add << " counts\n";

    countGate(name.str(), controlIndices, targetIndices, to_add);
    forwardWires(flow->inputs, wires);
    to_erase.insert(op);
    return true;
  }

  void preprocessOp(Operation *op, size_t to_add = 1) {
    if (preCount(op, to_add))
      return;

    if (auto loop = dyn_cast<cudaq::cc::LoopOp>(op)) {
      cudaq::opt::LoopComponents comp;
      if (cudaq::opt::isaInvariantLoop(loop, true, false, &comp)) {
        auto loopSize = comp.getIterationsConstant();
        if (!loopSize.has_value())
          return;
        auto iterations = loopSize.value();
        for (auto &b : loop.getBodyRegion().getBlocks())
          for (auto &op : b.getOperations())
            preprocessOp(&op, to_add * iterations);
      }
    } else if (auto ifop = dyn_cast<cudaq::cc::IfOp>(op)) {
      auto cond = ifop.getCondition();
      auto defop = cond.getDefiningOp();
      if (auto cop = dyn_cast<mlir::arith::ConstantOp>(defop)) {
        if (auto value = dyn_cast<BoolAttr>(cop.getValue())) {
          auto &region = value ? ifop.getThenRegion() : ifop.getElseRegion();
          for (auto &b : region.getBlocks())
            for (auto &op : b.getOperations())
              preprocessOp(&op, to_add);
        }
      }
    }
  }

  void runOnOperation() override {
    auto func = getOperation();

    for (auto &b : func.getBody()) {
      // We only pre-process the main block as the other blocks may be
      // conditional when the IR is lowered to CFG.
      if (&b != &func.getBody().front())
        continue;
      for (auto &op : b.getOperations())
        preprocessOp(&op);
    }
    for (auto op : to_erase)
      op->erase();

    to_erase.clear();
    qubitIndexMap.clear();
    nextQubitIndex = 0;
  }
};
