/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "UnitaryBuilder.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

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

static cl::opt<bool>
    dontCanonicalize("no-canonicalizer",
                     cl::desc("Disable running the canonicalizer pass."),
                     cl::init(false));

static cl::opt<bool> printUnitary("print-unitary",
                                  cl::desc("Print the unitary of each circuit"),
                                  cl::init(false));

static LogicalResult computeUnitary(func::FuncOp func,
                                    cudaq::UnitaryBuilder::UMatrix &unitary) {
  cudaq::UnitaryBuilder builder(unitary);
  return builder.build(func);
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  MLIRContext context;
  context.loadDialect<cudaq::cc::CCDialect, quake::QuakeDialect,
                      func::FuncDialect>();

  ParserConfig config(&context);
  auto checkMod = parseSourceFile<mlir::ModuleOp>(checkFilename, config);
  auto inputMod = parseSourceFile<mlir::ModuleOp>(inputFilename, config);

  // Run canonicalizer to make sure angles in parametrized quantum operations
  // are taking constants as inputs.
  if (!dontCanonicalize) {
    PassManager pm(&context);
    OpPassManager &nestedFuncPM = pm.nest<func::FuncOp>();
    nestedFuncPM.addPass(createCanonicalizerPass());
    if (failed(pm.run(*checkMod)) || failed(pm.run(*inputMod)))
      return EXIT_FAILURE;
  }

  auto applyTolerance = [](cudaq::UnitaryBuilder::UMatrix &m) {
    m = (1e-12 < m.array().abs()).select(m, 0.0f);
  };
  cudaq::UnitaryBuilder::UMatrix checkUnitary;
  cudaq::UnitaryBuilder::UMatrix inputUnitary;
  auto exitStatus = EXIT_SUCCESS;
  for (auto checkFunc : checkMod->getOps<func::FuncOp>()) {
    StringAttr opName = checkFunc.getSymNameAttr();
    checkUnitary.resize(0, 0);
    inputUnitary.resize(0, 0);
    // We need to check if input also has the same function
    auto *inputOp = inputMod->lookupSymbol(opName);
    assert(inputOp && "Function not present in input");

    auto inputFunc = dyn_cast<func::FuncOp>(inputOp);
    if (failed(computeUnitary(checkFunc, checkUnitary)) ||
        failed(computeUnitary(inputFunc, inputUnitary))) {
      llvm::errs() << "Cannot compute unitary for " << opName.str() << ".\n";
      exitStatus = EXIT_FAILURE;
      continue;
    }

    // Here we use std streams because Eigen printers don't work with LLVM ones.
    if (!cudaq::isApproxEqual(checkUnitary, inputUnitary, upToGlobalPhase)) {
      applyTolerance(checkUnitary);
      applyTolerance(inputUnitary);
      std::cerr << "Circuit: " << opName.str() << '\n';
      std::cerr << "Expected:\n";
      std::cerr << checkUnitary << '\n';
      std::cerr << "Got:\n";
      std::cerr << inputUnitary << '\n';
      exitStatus = EXIT_FAILURE;
    }

    if (printUnitary) {
      applyTolerance(checkUnitary);
      std::cout << "Circuit: " << opName.str() << '\n'
                << checkUnitary << "\n\n";
    }
  }
  return exitStatus;
}
