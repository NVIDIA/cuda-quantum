/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_EXPANDCONTROLVEQS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "Expand-control-veqs"

using namespace mlir;

namespace {
static Operation *replaceOperator(quake::OperatorInterface oldOp,
                                  ValueRange controls, OpBuilder &builder) {
  StringRef name = oldOp->getName().getStringRef();
  StringAttr nameAttr = builder.getStringAttr(name);
  ValueRange parameters = oldOp.getParameters();
  ValueRange targets = oldOp.getTargets();
  SmallVector<Value> operands(parameters);
  operands.append(controls.begin(), controls.end());
  operands.append(targets.begin(), targets.end());
  auto segmentSizes =
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(parameters.size()),
                                    static_cast<int32_t>(controls.size()),
                                    static_cast<int32_t>(targets.size())});
  auto newOp = builder.create(oldOp->getLoc(), nameAttr, operands);
  newOp->setAttr("operand_segment_sizes", segmentSizes);
  return newOp;
}

struct ExpandControlVeqsPass
    : public cudaq::opt::impl::ExpandControlVeqsBase<ExpandControlVeqsPass> {
  using ExpandControlVeqsBase::ExpandControlVeqsBase;

  LogicalResult expand(quake::OperatorInterface op) const {
    OpBuilder builder(op);
    SmallVector<Value> newControls;
    bool update = false;

    for (auto [index, control] : llvm::enumerate(op.getControls())) {
      size_t size = 1;
      if (auto veq = dyn_cast<quake::VeqType>(control.getType())) {
        if (!veq.hasSpecifiedSize())
          return failure();

        size = veq.getSize();
        for (size_t i = 0; i < size; ++i) {
          auto ext = builder.create<quake::ExtractRefOp>(op.getLoc(), control, i);
          newControls.push_back(ext);
          update = true;
        }
      } else {
        newControls.push_back(control);
      }
    }

    if (update) {
        replaceOperator(op, newControls, builder);
        op.erase();
    }

    return success();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.isExternal())
      return;

    func.walk([&](quake::OperatorInterface op) {
      // This pass must be run before conversion to Quake's value semantics form.
      if (!quake::isAllReferences(op))
        return;
      if (failed(expand(op)))
        return;
    });
  }
};

} // namespace