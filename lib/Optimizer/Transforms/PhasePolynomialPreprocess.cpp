/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "Subcircuit.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_PHASEPOLYNOMIALPREPROCESS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "phase-polynomial-preprocess"

using namespace mlir;

namespace {
class PhasePolynomialPreprocessPass : public cudaq::opt::impl::PhasePolynomialPreprocessBase<PhasePolynomialPreprocessPass> {
  using PhasePolynomialPreprocessBase::PhasePolynomialPreprocessBase;

    SetVector<Operation *> processed;
    SmallVector<Subcircuit> subcircuits;

public:
    // AXIS-SPECIFIC: could allow controlled y and z here
    bool isControlledOp(Operation *op) {
        return isa<quake::XOp>(op) && op->getNumOperands() == 2;
    }

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        
        func.walk([&](Operation *op) {
            if (!isControlledOp(op) || ::processed(op))
                return;

            Subcircuit subcircuit(op);
            subcircuits.push_back(subcircuit);
        });

        for (auto subcircuit : subcircuits) {
            llvm::outs() << "Calculated subcircuit: \n";
            for (auto *op : subcircuit.getOps())
                op->dump();
            llvm::outs() << "\n";
        }
    }
};
}