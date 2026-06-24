/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include <string>

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKEADDREQUIREDQUBITS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "quake-add-required-qubits"

using namespace mlir;

namespace {
/// Annotate a value-semantics Quake function with the number of physical qubits
/// it requires. The count is the highest `quake.borrow_wire` identity plus one,
/// stored as a string attribute (named `requiredQubits` by default;
/// configurable via the `attribute` option). A function that borrows no wires
/// is left unchanged.
///
/// The pass requires value semantics: every quantum gate
/// (`quake.OperatorInterface`) must operate on wires. A gate applied directly
/// to a reference (`!quake.ref`/`!quake.veq`, i.e. memory semantics) is
/// rejected.
class QuakeAddRequiredQubitsPass
    : public cudaq::opt::impl::QuakeAddRequiredQubitsBase<
          QuakeAddRequiredQubitsPass> {
public:
  using QuakeAddRequiredQubitsBase::QuakeAddRequiredQubitsBase;

  void runOnOperation() override {
    auto func = getOperation();

    // This pass requires value semantics. Every quantum gate
    // (OperatorInterface) must operate on wires; a gate applied directly to a
    // `!quake.ref` or `!quake.veq` is reference (memory) semantics and is
    // rejected.
    auto result = func.walk([](cudaq::quake::OperatorInterface op) {
      if (!cudaq::quake::isAllValues(op)) {
        op.emitOpError("quake-add-required-qubits requires value semantics");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    if (auto highest =
            cudaq::quake::getMaxBorrowedWireIndex(func.getOperation()))
      func->setAttr(
          attributeName,
          StringAttr::get(&getContext(), std::to_string(*highest + 1)));
  }
};
} // namespace
