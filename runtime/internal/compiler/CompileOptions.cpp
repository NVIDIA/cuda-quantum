/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq_internal/compiler/CompileOptions.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Transforms/AddMetadata.h"
#include "cudaq/runtime/logger/logger.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

/// Set the conditional measurement flag in the execution context if the module
/// contains conditional feedback.
static void setConditionalMeasurementFlag(cudaq::ExecutionContext &ctx,
                                          mlir::ModuleOp moduleOp) {
  for (auto &artifact : moduleOp) {
    quake::detail::QuakeFunctionAnalysis analysis{&artifact};
    auto info = analysis.getAnalysisInfo();
    if (info.empty())
      continue;

    auto result = info[&artifact];
    if (result.hasConditionalsOnMeasure) {
      ctx.hasConditionalsOnMeasureResults = true;
      break;
    }
  }
}

/// Warn the user if the kernel uses named measurement results in sampling mode.
static void warnNamedMeasurements(cudaq::ExecutionContext &ctx,
                                  mlir::ModuleOp moduleOp) {
  if (ctx.name != "sample" || ctx.warnedNamedMeasurements)
    return;

  auto funcOp = moduleOp.template lookupSymbol<mlir::func::FuncOp>(
      std::string(cudaq::runtime::cudaqGenPrefixName) + ctx.kernelName);
  if (!funcOp)
    return;

  bool hasNamedMeasurements = false;
  funcOp.walk([&](quake::MeasurementInterface meas) {
    if (meas.getOptionalRegisterName().has_value()) {
      hasNamedMeasurements = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (hasNamedMeasurements) {
    ctx.warnedNamedMeasurements = true;
    CUDAQ_WARN("Kernel {} uses named measurement results "
               "but is invoked in sampling mode. Support for "
               "sub-registers in `sample_result` is deprecated and will "
               "be removed in a future release. Use `run` to retrieve "
               "individual measurement results.",
               ctx.kernelName);
  }
}

void cudaq_internal::compiler::populateContextFromModule(
    cudaq::ExecutionContext &ctx, mlir::ModuleOp &moduleOp) {
  setConditionalMeasurementFlag(ctx, moduleOp);
  warnNamedMeasurements(ctx, moduleOp);
}
