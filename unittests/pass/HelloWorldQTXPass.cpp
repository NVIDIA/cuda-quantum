/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Support/Plugin.h"
#include "llvm/Analysis/CallGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include <cassert>
#include <iostream>

using namespace mlir;

namespace {

class HelloWorldQTXPass
    : public PassWrapper<HelloWorldQTXPass, OperationPass<qtx::CircuitOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HelloWorldQTXPass)

  llvm::StringRef getArgument() const override {
    return "cudaq-hello-world-qtx";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    llvm::errs() << "-- dump the circuit\n";
    circuit.dump();
  }
};

} // namespace

CUDAQ_REGISTER_MLIR_PASS(HelloWorldQTXPass)
