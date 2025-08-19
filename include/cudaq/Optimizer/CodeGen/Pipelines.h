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
                                mlir::StringRef codeGenFor = "qir:0.1",
                                mlir::StringRef passConfigAs = "qir:0.1");

/// \brief Pipeline builder to convert Quake to QIR.
/// Specifies a particular QIR profile in \p convertTo.
/// \p pm Pass manager to append passes to
/// \p convertTo QIR triple
/// The QIR triple is a name indicating the selected profile (`qir`, `qir-full`,
/// `qir-base`, or `qir-adaptive`) followed by an optional `:` and QIR version
/// followed by an optional `:` and a list of suboptions.
void addPipelineConvertToQIR(mlir::PassManager &pm, mlir::StringRef convertTo);

void addLowerToCCPipeline(mlir::OpPassManager &pm);

void addPipelineTranslateToOpenQASM(mlir::PassManager &pm);
void addPipelineTranslateToIQMJson(mlir::PassManager &pm);

} // namespace cudaq::opt
