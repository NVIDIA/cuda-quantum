//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//

//===- FabricInlineRealisations.cpp - normalize circuit realizations ------===//

#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/Fabric/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace qlx {
namespace fabric {
#define GEN_PASS_DEF_FABRICINLINEREALISATIONS
#include "qlx/Dialect/Fabric/Transforms/Passes.h.inc"
} // namespace fabric
} // namespace qlx

using namespace mlir;
using namespace qlx::fabric;

namespace {

class FabricInlineRealisationsPass
    : public qlx::fabric::impl::FabricInlineRealisationsBase<
          FabricInlineRealisationsPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    SmallVector<GadgetOp> worklist;
    module.walk([&](GadgetOp gadget) {
      if (gadget.getRealizationAttr())
        worklist.push_back(gadget);
    });

    for (GadgetOp gadget : worklist) {
      auto reference = gadget.getRealizationAttr();
      auto circuit = symbols.lookup<CircuitOp>(reference.getValue());
      if (!circuit || circuit.getBody().empty()) {
        gadget.emitError("realization '")
            << reference.getValue()
            << "' does not resolve to a fabric.circuit body";
        return signalPassFailure();
      }

      Block &gadgetBlock = gadget.getBody().front();
      Block &circuitBlock = circuit.getBody().front();
      IRMapping mapping;
      for (auto [source, destination] :
           llvm::zip(circuitBlock.getArguments(), gadgetBlock.getArguments()))
        mapping.map(source, destination);

      gadgetBlock.getTerminator()->erase();
      OpBuilder builder(&gadgetBlock, gadgetBlock.end());
      for (Operation &operation : circuitBlock)
        builder.clone(operation, mapping);
      gadget->removeAttr("realization");
      gadget->setAttr("realization_kind", builder.getStringAttr("inline"));
    }
  }
};

} // namespace
