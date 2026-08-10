/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_MULTICONTROLDECOMPOSITION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Operation *createOperator(Location loc, StringRef name,
                                 ValueRange parameters, ValueRange controls,
                                 ValueRange targets, OpBuilder &builder) {
  StringAttr nameAttr = builder.getStringAttr(name);
  SmallVector<Value> operands(parameters);
  operands.append(controls.begin(), controls.end());
  operands.append(targets.begin(), targets.end());
  auto segmentSizes = builder.getDenseI32ArrayAttr(
      {static_cast<std::int32_t>(parameters.size()),
       static_cast<std::int32_t>(controls.size()),
       static_cast<std::int32_t>(targets.size())});
  auto op = builder.create(loc, nameAttr, operands);
  op->setAttr("operand_segment_sizes", segmentSizes);
  return op;
}

//===----------------------------------------------------------------------===//
// Decomposer
//===----------------------------------------------------------------------===//

namespace {

class Decomposer {
public:
  Decomposer(func::FuncOp func) : builder(func) {
    entryBlock = &(*func.getBody().begin());
  }

  LogicalResult v_decomposition(cudaq::quake::OperatorInterface op);
  LogicalResult barenco_decomposition(cudaq::quake::OperatorInterface op);

private:
  LogicalResult extractControls(cudaq::quake::OperatorInterface op,
                                SmallVectorImpl<Value> &newControls,
                                SmallVectorImpl<bool> &negatedControls);

  ArrayRef<Value> getAncillas(Location loc, std::size_t numAncillas);

  void emitX(Location loc, ValueRange controls, Value target);
  void emitDirtyLadder(Location loc, ArrayRef<Value> controls,
                       ArrayRef<Value> dirty, Value target);
  void emitMultiControlX(Location loc, ArrayRef<Value> controls,
                         ArrayRef<Value> dirty, Value target);
  void emitBorrowedX(Location loc, ArrayRef<Value> controls, Value ancilla,
                     Value target);

  OpBuilder builder;
  Block *entryBlock;
  SmallVector<Value> allocatedAncillas;
};

} // namespace

LogicalResult
Decomposer::extractControls(cudaq::quake::OperatorInterface op,
                            SmallVectorImpl<Value> &newControls,
                            SmallVectorImpl<bool> &negatedControls) {
  auto negControls = op.getNegatedControls();
  for (auto [index, control] : llvm::enumerate(op.getControls())) {
    size_t size = 1;
    if (isa<cudaq::quake::RefType>(control.getType())) {
      newControls.push_back(control);
    } else if (auto veq = dyn_cast<cudaq::quake::VeqType>(control.getType())) {
      if (!veq.hasSpecifiedSize())
        return failure();
      size = veq.getSize();
      for (size_t i = 0; i < size; ++i)
        newControls.push_back(cudaq::quake::ExtractRefOp::create(
            builder, op.getLoc(), control, i));
    }
    if (negControls)
      negatedControls.append(size, (*negControls)[index]);
  }
  return success();
}

ArrayRef<Value> Decomposer::getAncillas(Location loc, std::size_t numAncillas) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(entryBlock);
  // If we don't have enough ancillas, allocate some more. The allocations land
  // at the top of the entry block, so they do not follow the kernel's own
  // qubits; mark them instead of relying on that order.
  for (size_t i = allocatedAncillas.size(); i < numAncillas; ++i) {
    auto alloca = cudaq::quake::AllocaOp::create(builder, loc);
    cudaq::quake::markAsAncilla(alloca);
    allocatedAncillas.push_back(alloca);
  }
  return {allocatedAncillas.begin(), allocatedAncillas.begin() + numAncillas};
}

//===----------------------------------------------------------------------===//
// Barenco decomposition
//
// A constant number of extra qubits instead of one per control, paid for in
// gates. The idea (Barenco et al., "Elementary gates for quantum computation",
// Lemmas 7.2 and 7.3) is that the operation's own controls can serve as
// scratch space: they are read-only for the operation being decomposed, and a
// ladder that touches them can be built to restore them exactly, whatever
// state they are in.
//===----------------------------------------------------------------------===//

void Decomposer::emitX(Location loc, ValueRange controls, Value target) {
  cudaq::quake::XOp::create(builder, loc, controls, target);
}

/// Emit a `controls.size()`-controlled X on \p target using \p dirty as
/// scratch. The scratch qubits may be in any state and are left exactly as
/// they were found; \p dirty must hold at least `controls.size() - 2` qubits,
/// none of them aliasing a control or the target.
///
/// The circuit is two identical sweeps. Each sweep walks the ladder down to
/// the innermost pair and back up. A single sweep leaves every ancilla holding
/// garbage; running it twice cancels that garbage while the flips on the
/// target, which each sweep applies once, reinforce.
void Decomposer::emitDirtyLadder(Location loc, ArrayRef<Value> controls,
                                 ArrayRef<Value> dirty, Value target) {
  const int n = controls.size();
  assert(n >= 3 && static_cast<int>(dirty.size()) >= n - 2);

  auto sweep = [&]() {
    emitX(loc, {controls[n - 1], dirty[n - 3]}, target);
    for (int i = n - 3; i >= 1; --i)
      emitX(loc, {controls[i + 1], dirty[i - 1]}, dirty[i]);
    emitX(loc, {controls[0], controls[1]}, dirty[0]);
    for (int i = 1; i <= n - 3; ++i)
      emitX(loc, {controls[i + 1], dirty[i - 1]}, dirty[i]);
  };
  sweep();
  sweep();
}

/// Emit a multi-controlled X, picking the cheapest form for the control count.
void Decomposer::emitMultiControlX(Location loc, ArrayRef<Value> controls,
                                   ArrayRef<Value> dirty, Value target) {
  if (controls.size() <= 2) {
    emitX(loc, controls, target);
    return;
  }
  emitDirtyLadder(loc, controls, dirty.take_front(controls.size() - 2), target);
}

/// Emit a multi-controlled X on \p target using exactly one clean ancilla.
///
/// The controls are split in half. One half is `AND`-ed into the ancilla,
/// borrowing the other half (and the target) as dirty scratch; the result
/// controls the rest; then the `AND` is undone. Each half is small enough that
/// the other half covers its scratch needs, which is what keeps the clean
/// ancilla count at one however many controls there are.
void Decomposer::emitBorrowedX(Location loc, ArrayRef<Value> controls,
                               Value ancilla, Value target) {
  if (controls.size() <= 2) {
    emitX(loc, controls, target);
    return;
  }
  const std::size_t half = (controls.size() + 1) / 2;
  ArrayRef<Value> lower = controls.take_front(half);
  ArrayRef<Value> upper = controls.drop_front(half);

  SmallVector<Value> lowerScratch(upper);
  lowerScratch.push_back(target);
  SmallVector<Value> upperControls(upper);
  upperControls.push_back(ancilla);

  emitMultiControlX(loc, lower, lowerScratch, ancilla);
  emitMultiControlX(loc, upperControls, lower, target);
  emitMultiControlX(loc, lower, lowerScratch, ancilla);
}

LogicalResult
Decomposer::barenco_decomposition(cudaq::quake::OperatorInterface op) {
  builder.setInsertionPoint(op);
  SmallVector<Value> controls;
  SmallVector<bool> negatedControls;
  if (failed(extractControls(op, controls, negatedControls)))
    return failure();

  if (controls.size() <= 1)
    return failure();
  if (controls.size() == 2 && isa<cudaq::quake::XOp, cudaq::quake::ZOp>(op))
    return failure();

  Location loc = op->getLoc();
  StringRef name = op->getName().getStringRef();
  ValueRange parameters = op.getParameters();
  ValueRange targets = op.getTargets();

  // A negated control is an ordinary control conjugated by X. Doing it here
  // once keeps every construction below free of negation bookkeeping.
  auto flipNegatedControls = [&]() {
    for (auto [isNegated, control] : llvm::zip(negatedControls, controls))
      if (isNegated)
        cudaq::quake::XOp::create(builder, loc, ValueRange{}, control);
  };
  flipNegatedControls();

  if (isa<cudaq::quake::XOp, cudaq::quake::ZOp>(op)) {
    Value target = targets.front();
    // C^n(Z) is C^n(X) conjugated by H on the target.
    const bool isZ = isa<cudaq::quake::ZOp>(op);
    if (isZ)
      cudaq::quake::HOp::create(builder, loc, ValueRange{}, target);
    emitBorrowedX(loc, controls, getAncillas(loc, 1).front(), target);
    if (isZ)
      cudaq::quake::HOp::create(builder, loc, ValueRange{}, target);
  } else {
    // Everything else needs the conjunction of the controls in hand before the
    // operation can be applied singly-controlled: one ancilla to hold it, and
    // (beyond two controls) one more as scratch to compute it with.
    auto ancillas = getAncillas(loc, controls.size() <= 2 ? 1 : 2);
    Value conjunction = ancillas.front();
    Value scratch = ancillas.back();
    emitBorrowedX(loc, controls, scratch, conjunction);
    createOperator(loc, name, parameters, conjunction, targets, builder);
    emitBorrowedX(loc, controls, scratch, conjunction);
  }

  flipNegatedControls();
  return success();
}

LogicalResult Decomposer::v_decomposition(cudaq::quake::OperatorInterface op) {
  builder.setInsertionPoint(op);
  // First, we need to extract controls from any `veq` that might been used as
  // a control for this operation.
  SmallVector<Value> controls;
  SmallVector<bool> negatedControls;
  if (failed(extractControls(op, controls, negatedControls)))
    return failure();

  // We only decompose operations with multiple controls.
  if (controls.size() <= 1)
    return failure();

  // We don't decompose CCX and CCZ as they are handle by another pass.
  if (controls.size() == 2 && isa<cudaq::quake::XOp, cudaq::quake::ZOp>(op))
    return failure();

  // Operator info
  Location loc = op->getLoc();
  StringRef name = op->getName().getStringRef();
  ValueRange parameters = op.getParameters();
  ValueRange targets = op.getTargets();

  // Compute the required number of ancillas to decompose this operation.
  // Allocate new qubits if necessary.
  size_t requiredAncillas = isa<cudaq::quake::XOp, cudaq::quake::ZOp>(op)
                                ? controls.size() - 2
                                : controls.size() - 1;
  auto ancillas = getAncillas(loc, requiredAncillas);

  // Compute intermediate results
  SmallVector<Operation *> toCleanup;
  std::array<Value, 2> cs = {controls[0], controls[1]};
  toCleanup.push_back(cudaq::quake::XOp::create(builder, loc, cs, ancillas[0]));
  if (!negatedControls.empty() && (negatedControls[0] || negatedControls[1]))
    toCleanup.back()->setAttr("negated_qubit_controls",
                              builder.getDenseBoolArrayAttr(
                                  {negatedControls[0], negatedControls[1]}));
  for (std::size_t c = 2, a = 0, n = requiredAncillas + 1; c < n; ++c, ++a) {
    cs = {controls[c], ancillas[a]};
    toCleanup.push_back(
        cudaq::quake::XOp::create(builder, loc, cs, ancillas[a + 1]));
    if (!negatedControls.empty() && negatedControls[c])
      toCleanup.back()->setAttr("negated_qubit_controls",
                                builder.getDenseBoolArrayAttr({true, false}));
  }

  // Compute output
  if (!isa<cudaq::quake::XOp, cudaq::quake::ZOp>(op)) {
    createOperator(loc, name, parameters, ancillas.back(), targets, builder);
  } else {
    cs = {controls.back(), ancillas.back()};
    Operation *out =
        createOperator(loc, name, parameters, cs, targets, builder);
    if (!negatedControls.empty() && negatedControls.back())
      out->setAttr("negated_qubit_controls",
                   builder.getDenseBoolArrayAttr({true, false}));
  }

  // Cleanup intermediate results
  for (Operation *op : llvm::reverse(toCleanup))
    builder.clone(*op);

  return success();
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//
namespace {
struct Decomposition
    : public cudaq::opt::impl::MultiControlDecompositionBase<Decomposition> {
  using MultiControlDecompositionBase::MultiControlDecompositionBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.isExternal())
      return;

    const bool useBarenco = strategy == "barenco";
    if (!useBarenco && strategy != "v-ladder") {
      func.emitError("unknown multicontrol-decomposition strategy '" +
                     strategy + "'; expected 'v-ladder' or 'barenco'");
      signalPassFailure();
      return;
    }

    Decomposer decomposer(func);
    func.walk([&](cudaq::quake::OperatorInterface op) {
      // This pass does not handle Quake's value semantics form.
      if (!cudaq::quake::isAllReferences(op))
        return;
      if (failed(useBarenco ? decomposer.barenco_decomposition(op)
                            : decomposer.v_decomposition(op)))
        return;
      op.erase();
    });
  }
};
} // namespace
