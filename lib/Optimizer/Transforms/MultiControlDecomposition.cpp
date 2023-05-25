/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

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

namespace {

Operation *createOperator(Location loc, StringRef name, ValueRange parameters,
                          ValueRange controls, ValueRange targets,
                          OpBuilder &builder) {
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

LogicalResult extractControls(quake::OperatorInterface op,
                              SmallVectorImpl<Value> &newControls,
                              SmallVectorImpl<bool> &negatedControls,
                              OpBuilder &builder) {
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

void v_decomposition(quake::OperatorInterface op, ValueRange controls,
                     ArrayRef<bool> negatedControls, OpBuilder &builder) {
  // Operator info
  Location loc = op->getLoc();
  StringRef name = op->getName().getStringRef();
  ValueRange parameters = op.getParameters();
  ValueRange targets = op.getTargets();
  const size_t numControlsOutput = isa<quake::XOp, quake::ZOp>(op) ? 2 : 1;

  SmallVector<Value> ancillas;
  for (size_t i = 0, n = controls.size() - numControlsOutput; i < n; ++i)
    ancillas.push_back(builder.create<quake::AllocaOp>(loc));

  SmallVector<Operation *> toCleanup;
  std::array<Value, 2> cs = {controls[0], controls[1]};

  // Compute intermediate results
  toCleanup.push_back(builder.create<quake::XOp>(loc, cs, ancillas[0]));
  if (!negatedControls.empty() &&
      (negatedControls[0] != false || negatedControls[0] != false))
    toCleanup.back()->setAttr("negated_qubit_controls",
                              builder.getDenseBoolArrayAttr(
                                  {negatedControls[0], negatedControls[1]}));
  for (std::size_t c = 2, a = 0, n = controls.size() - numControlsOutput + 1; c < n;
       ++c, ++a) {
    cs = {controls[c], ancillas[a]};
    toCleanup.push_back(builder.create<quake::XOp>(loc, cs, ancillas[a + 1]));
    if (!negatedControls.empty() && negatedControls[c] != false)
      toCleanup.back()->setAttr("negated_qubit_controls",
                                builder.getDenseBoolArrayAttr({true, false}));
  }

  // Compute output
  if (numControlsOutput == 1) {
    createOperator(loc, name, parameters, ancillas.back(), targets, builder);
  } else {
    cs = {controls.back(), ancillas.back()};
    Operation *out =
        createOperator(loc, name, parameters, cs, targets, builder);
    if (!negatedControls.empty() && negatedControls.back() != false)
      out->setAttr("negated_qubit_controls",
                   builder.getDenseBoolArrayAttr({true, false}));
  }

  // Cleanup intermediate results
  for (Operation *op : llvm::reverse(toCleanup))
    builder.clone(*op);
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct Decomposition
    : public opt::impl::MultiControlDecompositionPassBase<Decomposition> {
  using MultiControlDecompositionPassBase::MultiControlDecompositionPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    OpBuilder builder(func);
    func.walk([&](quake::OperatorInterface op) {
      // This pass does not handle Quake's value semantics form.
      if (!quake::isAllReferences(op))
        return;

      builder.setInsertionPoint(op);
      SmallVector<Value> controls;
      SmallVector<bool> negatedControls;
      if (failed(extractControls(op, controls, negatedControls, builder)))
        return;

      if (controls.size() == 2) {
        if (isa<quake::XOp, quake::ZOp>(op))
          return;
        Value ancilla = builder.create<quake::AllocaOp>(op->getLoc());
        auto andOp =
            builder.create<quake::XOp>(op->getLoc(), controls, ancilla);
        andOp.setNegatedQubitControls(op.getNegatedControls());
        createOperator(op->getLoc(), op->getName().getStringRef(),
                       op.getParameters(), ancilla, op.getTargets(), builder);
        builder.clone(*andOp);
        op.erase();
        return;
      }
      v_decomposition(op, controls, negatedControls, builder);
      op.erase();
    });
  }
};

} // namespace
