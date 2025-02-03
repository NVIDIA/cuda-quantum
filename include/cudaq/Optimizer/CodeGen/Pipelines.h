/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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

/// Adds the common pipeline. \p codeGenFor specifies which variant of QIR is to
/// be generated: full, base-profile, adaptive-profile, etc. \p passConfigAs
/// specifies which variant of QIR to use with \e other passes, and not the
/// final `codegen`, in the pipeline. Typically, \p codeGenFor and \p
/// passConfigAs will have identical values.
void commonPipelineConvertToQIR(mlir::PassManager &pm,
                                mlir::StringRef codeGenFor = "qir",
                                mlir::StringRef passConfigAs = "qir");

/// \deprecated{Only for Python, since it can't use the new QIR codegen.}
void commonPipelineConvertToQIR_PythonWorkaround(
    mlir::PassManager &pm, const std::optional<mlir::StringRef> &convertTo);

/// \brief Pipeline builder to convert Quake to QIR.
/// Does not specify a particular QIR profile.
inline void addPipelineConvertToQIR(mlir::PassManager &pm) {
  commonPipelineConvertToQIR(pm);
}

/// \deprecated{Only for Python, since it can't use the new QIR codegen.}
inline void addPipelineConvertToQIR_PythonWorkaround(mlir::PassManager &pm) {
  commonPipelineConvertToQIR_PythonWorkaround(pm, std::nullopt);
}

/// \brief Pipeline builder to convert Quake to QIR.
/// Specifies a particular QIR profile in \p convertTo.
/// \p pm Pass manager to append passes to
/// \p convertTo name of QIR profile (e.g., `qir-base`, `qir-adaptive`, ...)
void addPipelineConvertToQIR(mlir::PassManager &pm, mlir::StringRef convertTo);

/// \deprecated{Only for Python, since it can't use the new QIR codegen.}
inline void
addPipelineConvertToQIR_PythonWorkaround(mlir::PassManager &pm,
                                         mlir::StringRef convertTo) {
  commonPipelineConvertToQIR_PythonWorkaround(pm, convertTo);
  addQIRProfilePipeline(pm, convertTo);
}

void addLowerToCCPipeline(mlir::OpPassManager &pm);

void addPipelineTranslateToOpenQASM(mlir::PassManager &pm);
void addPipelineTranslateToIQMJson(mlir::PassManager &pm);

} // namespace cudaq::opt
