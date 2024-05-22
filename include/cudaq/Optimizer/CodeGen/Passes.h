/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// \file
/// The OptCodeGen library includes passes that lower the MLIR module for some
/// particular quantum target representation. There is a bevy of such targets
/// that provide platforms on which the quantum code can be run.

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class MLIRContext;
class Pass;
class PassManager;
} // namespace mlir

namespace cudaq::opt {

std::unique_ptr<mlir::Pass> createConvertToQIRPass();
void registerConvertToQIRPass();

/// Convert (generic) QIR to the profile-specific QIR for a specific target.
/// @param pm Pass Manager to add QIR passes to
/// @param convertTo Expected to be `qir-base` or `qir-adaptive` (comes from the
/// cudaq-translate command line `--convert-to` parameter)
void addQIRProfilePipeline(mlir::OpPassManager &pm, llvm::StringRef convertTo);

/// @brief Verify that all `CallOp` targets are QIR- or NVQIR-defined functions
/// or in the provided allowed list.
std::unique_ptr<mlir::Pass>
createVerifyNVQIRCallOpsPass(const std::vector<llvm::StringRef> &allowedFuncs);

// Use the addQIRProfilePipeline() for the following passes.
std::unique_ptr<mlir::Pass>
createQIRToQIRProfilePass(llvm::StringRef convertTo);
std::unique_ptr<mlir::Pass> verifyQIRProfilePass(llvm::StringRef convertTo);
std::unique_ptr<mlir::Pass> createQIRProfilePreparationPass();
std::unique_ptr<mlir::Pass>
createConvertToQIRFuncPass(llvm::StringRef convertTo);

/// Register target pipelines.
void registerTargetPipelines();

/// Register CodeGenDialect with the provided DialectRegistry.
void registerCodeGenDialect(mlir::DialectRegistry &registry);

// declarative passes
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"

} // namespace cudaq::opt
