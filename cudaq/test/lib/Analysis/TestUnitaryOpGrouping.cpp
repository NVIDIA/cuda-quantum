/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/UnitaryOpGrouping.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
class TestUnitaryOpGroupingPass
    : public PassWrapper<TestUnitaryOpGroupingPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestUnitaryOpGroupingPass)

  StringRef getArgument() const final { return "test-unitary-op-grouping"; }

  StringRef getDescription() const final {
    return "Print found groups of unitary operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      auto &analysis = getChildAnalysis<
          cudaq::quake::detail::UnitaryOpGroupingAnalysis>(
          funcOp.getOperation());

      llvm::errs() << "func @" << funcOp.getName() << '\n';

      unsigned groupIndex = 0;
      for (const auto &group : analysis.getGroups()) {
        llvm::errs() << "  unitary group #" << groupIndex++ << " -- found "
                     << group.ops.size() << " ops\n";
        for (Operation *op : group.ops)
          llvm::errs() << "    " << op->getName().getStringRef() << '\n';
      }
    }

    markAllAnalysesPreserved();
  }
};
} // namespace

namespace cudaq::test {
void registerTestUnitaryOpGroupingPass() {
  PassRegistration<TestUnitaryOpGroupingPass>();
}
} // namespace cudaq::test
