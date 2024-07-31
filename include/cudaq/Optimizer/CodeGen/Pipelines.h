/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file
/// Define some pipeline instantiation functions that can be shared between
/// the various tools and the runtime.

#pragma once

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Conversion/MathToFuncs/MathToFuncs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {

/// Adds the common pipeline lowering passes.
void commonLoweringPipeline(
    mlir::PassManager &pm, const mlir::StringRef &gateset, const std::optional<mlir::StringRef> &convertTo);

/// The common pipeline.
/// Adds the common pipeline (with or without a profile specifier) but without
/// the final QIR profile lowering passes.
void commonPipelineConvertToQIR(
    mlir::PassManager &pm, const std::optional<mlir::StringRef> &convertTo, const std::optional<mlir::StringRef> &mapping);

/// \brief Pipeline builder to convert Quake to QIR.
/// Does not specify a particular QIR profile.
inline void addPipelineConvertToQIR(mlir::PassManager &pm) {
  commonPipelineConvertToQIR(pm, std::nullopt, std::nullopt);
}

/// \brief Pipeline builder to convert Quake to QIR.
/// Specifies a particular QIR profile in \p convertTo.
/// \p pm Pass manager to append passes to
/// \p convertTo name of QIR profile (e.g., `qir-base`, `qir-adaptive`, ...)
inline void addPipelineConvertToQIR(mlir::PassManager &pm,
                                    mlir::StringRef convertTo,
                                    const std::optional<mlir::StringRef> &mapping) {
  commonPipelineConvertToQIR(pm, convertTo, mapping);
  addQIRProfilePipeline(pm, convertTo);
}

void addLowerToCCPipeline(mlir::OpPassManager &pm);

void addPipelineTranslateToOpenQASM(mlir::PassManager &pm, const std::optional<mlir::StringRef> &mapping);
void addPipelineTranslateToIQMJson(mlir::PassManager &pm, const std::optional<mlir::StringRef> &mapping);

} // namespace cudaq::opt
