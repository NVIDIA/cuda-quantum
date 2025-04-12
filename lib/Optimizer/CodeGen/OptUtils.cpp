//===- OptUtils.cpp - MLIR Execution Engine optimization pass utilities ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Additional modifications by NVIDIA Corporation.
// - Use cudaq namespace instead of mlir namespace.
// - Remove usage of formatv for error message (convenience)

//===----------------------------------------------------------------------===//
//
// This file implements the utility functions to trigger LLVM optimizations from
// MLIR Execution Engine.
//
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/CodeGen/OptUtils.h"

#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include <optional>

using namespace llvm;

static std::optional<OptimizationLevel> mapToLevel(unsigned optLevel,
                                                   unsigned sizeLevel) {
  switch (optLevel) {
  case 0:
    return OptimizationLevel::O0;

  case 1:
    return OptimizationLevel::O1;

  case 2:
    switch (sizeLevel) {
    case 0:
      return OptimizationLevel::O2;

    case 1:
      return OptimizationLevel::Os;

    case 2:
      return OptimizationLevel::Oz;
    }
    break;
  case 3:
    return OptimizationLevel::O3;
  }
  return std::nullopt;
}
// Create and return a lambda that uses LLVM pass manager builder to set up
// optimizations based on the given level.
std::function<Error(Module *)>
cudaq::makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel,
                                 TargetMachine *targetMachine,
                                 bool allowVectorization) {
  return [optLevel, sizeLevel, targetMachine,
          allowVectorization](Module *m) -> Error {
    std::optional<OptimizationLevel> ol = mapToLevel(optLevel, sizeLevel);
    if (!ol) {
      return make_error<StringError>(
          std::string("invalid optimization/size level"),
          inconvertibleErrorCode());
    }
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    tuningOptions.LoopVectorization = allowVectorization;
    tuningOptions.SLPVectorization = allowVectorization;

    PassBuilder pb(targetMachine, tuningOptions);

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    if (*ol == OptimizationLevel::O0)
      mpm.addPass(pb.buildO0DefaultPipeline(*ol));
    else
      mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));

    mpm.run(*m, mam);
    return Error::success();
  };
}
