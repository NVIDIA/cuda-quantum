/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "UnitaryBuilder.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"

#include <iostream>

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> checkFilename(cl::Positional,
                                          cl::desc("<check file>"));
static cl::opt<std::string>
    inputFilename("input", cl::desc("File to check (defaults to stdin)"),
                  cl::init("-"));

static cl::opt<bool>
    upToGlobalPhase("up-to-global-phase",
                    cl::desc("Check unitaries are equal up to global phase."),
                    cl::init(false));

static cl::opt<bool> printUnitary("print-unitary",
                                  cl::desc("Print the unitary of each circuit"),
                                  cl::init(false));

static LogicalResult computeUnitary(mlir::Operation *op,
                                    cudaq::UnitaryBuilder::UMatrix &unitary) {
  cudaq::UnitaryBuilder builder(unitary);
  if (auto func = dyn_cast_if_present<func::FuncOp>(op)) {
    auto a = builder.build(func);
    return a;
  }
  if (auto circuit = dyn_cast_if_present<qtx::CircuitOp>(op)) {
    return builder.build(circuit);
  }
  return failure();
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  MLIRContext context;
  context.loadDialect<qtx::QTXDialect, quake::QuakeDialect, func::FuncDialect,
                      memref::MemRefDialect>();

  mlir::ParserConfig config(&context);
  auto checkMod = mlir::parseSourceFile<mlir::ModuleOp>(checkFilename, config);
  auto inputMod = mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, config);

  cudaq::UnitaryBuilder::UMatrix checkUnitary;
  cudaq::UnitaryBuilder::UMatrix inputUnitary;
  for (auto &checkOp : checkMod->getBodyRegion().getOps()) {
    mlir::StringAttr opName;
    if (auto func = dyn_cast<func::FuncOp>(checkOp)) {
      opName = func.getSymNameAttr();
    } else if (auto circuit = dyn_cast_if_present<qtx::CircuitOp>(checkOp)) {
      opName = circuit.getSymNameAttr();
    } else {
      continue;
    }
    checkUnitary.resize(0, 0);
    inputUnitary.resize(0, 0);
    // We need to check if input also has the same function
    auto *inputOp = inputMod->lookupSymbol(opName);
    if (failed(computeUnitary(&checkOp, checkUnitary)) ||
        failed(computeUnitary(inputOp, inputUnitary)))
      continue;

    if (!cudaq::isApproxEqual(checkUnitary, inputUnitary, upToGlobalPhase)) {
      std::cerr << "Circuit: " << opName.str() << '\n';
      std::cerr << "Expected:\n";
      std::cerr << checkUnitary << '\n';
      std::cerr << "Got:\n";
      std::cerr << inputUnitary << '\n';
    }

    if (printUnitary) {
      std::cout << "Circuit: " << opName.str() << '\n'
                << checkUnitary << "\n\n";
      inputUnitary =
          (1e-12 < inputUnitary.array().abs()).select(inputUnitary, 0.0f);

      std::cout << "Unitary:\n" << inputUnitary << "\n\n";
    }
  }
  return 0;
}
