/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

using namespace mlir;
using namespace cudaq;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

namespace cudaq::opt {
#define GEN_PASS_DEF_MULTICONTROLDECOMPOSITIONPASS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

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
  auto segmentSizes =
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(parameters.size()),
                                    static_cast<int32_t>(controls.size()),
                                    static_cast<int32_t>(targets.size())});
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

  LogicalResult v_decomposition(quake::OperatorInterface op);

private:
  LogicalResult extractControls(quake::OperatorInterface op,
                                SmallVectorImpl<Value> &newControls,
                                SmallVectorImpl<bool> &negatedControls);

  ArrayRef<Value> getAncillas(Location loc, std::size_t numAncillas);

  OpBuilder builder;
  Block *entryBlock;
  SmallVector<Value> allocatedAncillas;
};

} // namespace

LogicalResult
Decomposer::extractControls(quake::OperatorInterface op,
                            SmallVectorImpl<Value> &newControls,
                            SmallVectorImpl<bool> &negatedControls) {
  auto negControls = op.getNegatedControls();
  for (auto [index, control] : llvm::enumerate(op.getControls())) {
    size_t size = 1;
    if (isa<quake::RefType>(control.getType())) {
      newControls.push_back(control);
    } else if (auto veq = dyn_cast<quake::VeqType>(control.getType())) {
      if (!veq.hasSpecifiedSize())
        return failure();
      size = veq.getSize();
      for (size_t i = 0; i < size; ++i)
        newControls.push_back(
            builder.create<quake::ExtractRefOp>(op.getLoc(), control, i));
    }
    if (negControls)
      negatedControls.append(size, (*negControls)[index]);
  }
  return success();
}

ArrayRef<Value> Decomposer::getAncillas(Location loc, std::size_t numAncillas) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(entryBlock);
  // If we don't have enough ancillas, allocate new more.
  for (size_t i = allocatedAncillas.size(); i < numAncillas; ++i)
    allocatedAncillas.push_back(builder.create<quake::AllocaOp>(loc));
  return {allocatedAncillas.begin(), allocatedAncillas.begin() + numAncillas};
}

LogicalResult Decomposer::v_decomposition(quake::OperatorInterface op) {
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
  if (controls.size() == 2 && isa<quake::XOp, quake::ZOp>(op))
    return failure();

  // Operator info
  Location loc = op->getLoc();
  StringRef name = op->getName().getStringRef();
  ValueRange parameters = op.getParameters();
  ValueRange targets = op.getTargets();

  // Compute the required number of ancillas to decompose this operation.
  // Allocate new qubits if necessary.
  size_t requiredAncillas = isa<quake::XOp, quake::ZOp>(op)
                                ? controls.size() - 2
                                : controls.size() - 1;
  auto ancillas = getAncillas(loc, requiredAncillas);

  // Compute intermediate results
  SmallVector<Operation *> toCleanup;
  std::array<Value, 2> cs = {controls[0], controls[1]};
  toCleanup.push_back(builder.create<quake::XOp>(loc, cs, ancillas[0]));
  if (!negatedControls.empty() && (negatedControls[0] || negatedControls[1]))
    toCleanup.back()->setAttr("negated_qubit_controls",
                              builder.getDenseBoolArrayAttr(
                                  {negatedControls[0], negatedControls[1]}));
  for (std::size_t c = 2, a = 0, n = requiredAncillas + 1; c < n; ++c, ++a) {
    cs = {controls[c], ancillas[a]};
    toCleanup.push_back(builder.create<quake::XOp>(loc, cs, ancillas[a + 1]));
    if (!negatedControls.empty() && negatedControls[c])
      toCleanup.back()->setAttr("negated_qubit_controls",
                                builder.getDenseBoolArrayAttr({true, false}));
  }

  // Compute output
  if (!isa<quake::XOp, quake::ZOp>(op)) {
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
    : public opt::impl::MultiControlDecompositionPassBase<Decomposition> {
  using MultiControlDecompositionPassBase::MultiControlDecompositionPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.isExternal())
      return;

    Decomposer decomposer(func);
    func.walk([&](quake::OperatorInterface op) {
      // This pass does not handle Quake's value semantics form.
      if (!quake::isAllReferences(op))
        return;
      if (failed(decomposer.v_decomposition(op)))
        return;
      op.erase();
    });
  }
};

} // namespace
