/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

// Command-line options
static cl::opt<std::string> checkFilename(cl::Positional, cl::desc("<check file>"));
static cl::opt<std::string> inputFilename("input", cl::desc("File to check (defaults to stdin)"), cl::init("-"));
static cl::opt<bool> upToGlobalPhase("up-to-global-phase", cl::desc("Check unitaries up to global phase."), cl::init(false));
static cl::opt<bool> upToMapping("up-to-mapping", cl::desc("Check unitaries with known permutation."), cl::init(false));
static cl::opt<bool> dontCanonicalize("no-canonicalizer", cl::desc("Disable canonicalization pass."), cl::init(false));
static cl::opt<bool> printUnitary("print-unitary", cl::desc("Print unitary matrices."), cl::init(false));

/**
 * Computes the unitary matrix for a given function.
 * @param func The function operation to analyze.
 * @param unitary Output matrix to store the computed unitary.
 * @param upToMapping Whether to allow permutations in computation.
 * @return LogicalResult indicating success or failure.
 */
static LogicalResult computeUnitary(func::FuncOp func, cudaq::UnitaryBuilder::UMatrix &unitary, bool upToMapping = false) {
    cudaq::UnitaryBuilder builder(unitary, upToMapping);
    return builder.build(func);
}

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv);

    // Initialize MLIR context and load required dialects
    MLIRContext context;
    context.loadDialect<cudaq::cc::CCDialect, quake::QuakeDialect, func::FuncDialect>();

    // Parse input and check files
    ParserConfig config(&context);
    auto checkMod = parseSourceFile<mlir::ModuleOp>(checkFilename, config);
    auto inputMod = parseSourceFile<mlir::ModuleOp>(inputFilename, config);
    if (!checkMod || !inputMod) {
        llvm::errs() << "Error parsing input files.\n";
        return EXIT_FAILURE;
    }

    // Apply canonicalization if enabled
    if (!dontCanonicalize) {
        PassManager pm(&context);
        OpPassManager &nestedFuncPM = pm.nest<func::FuncOp>();
        nestedFuncPM.addPass(createCanonicalizerPass());
        if (failed(pm.run(*checkMod)) || failed(pm.run(*inputMod))) {
            llvm::errs() << "Canonicalization failed.\n";
            return EXIT_FAILURE;
        }
    }

    // Apply tolerance to small numerical values in matrices
    auto applyTolerance = [](cudaq::UnitaryBuilder::UMatrix &m) {
        m = (1e-12 < m.array().abs()).select(m, 0.0f);
    };

    // Iterate through functions in check module and compare against input module
    int exitStatus = EXIT_SUCCESS;
    for (auto checkFunc : checkMod->getOps<func::FuncOp>()) {
        StringAttr opName = checkFunc.getSymNameAttr();
        cudaq::UnitaryBuilder::UMatrix checkUnitary, inputUnitary;

        // Look up corresponding function in input module
        auto *inputOp = inputMod->lookupSymbol(opName);
        if (!inputOp) {
            llvm::errs() << "Function " << opName.str() << " not found in input.\n";
            exitStatus = EXIT_FAILURE;
            continue;
        }

        auto inputFunc = dyn_cast<func::FuncOp>(inputOp);
        if (!inputFunc || failed(computeUnitary(checkFunc, checkUnitary)) ||
            failed(computeUnitary(inputFunc, inputUnitary, upToMapping))) {
            llvm::errs() << "Cannot compute unitary for " << opName.str() << ".\n";
            exitStatus = EXIT_FAILURE;
            continue;
        }

        // Compare unitaries and report differences
        if (!cudaq::isApproxEqual(checkUnitary, inputUnitary, upToGlobalPhase)) {
            applyTolerance(checkUnitary);
            applyTolerance(inputUnitary);
            std::cerr << "Circuit: " << opName.str() << '\n';
            std::cerr << "Expected:\n" << checkUnitary << '\n';
            std::cerr << "Got:\n" << inputUnitary << '\n';
            exitStatus = EXIT_FAILURE;
        }

        // Print unitaries if requested
        if (printUnitary) {
            applyTolerance(checkUnitary);
            std::cout << "Circuit: " << opName.str() << '\n' << checkUnitary << "\n\n";
        }
    }
    return exitStatus;
}
