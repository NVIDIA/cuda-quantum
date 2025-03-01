/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
class LLVMTypeConverter;
class MLIRContext;
class Pass;
class PassManager;
namespace LLVM {
class LLVMStructType;
}
} // namespace mlir

namespace cudaq::opt {

/// Convert (generic) QIR to the profile-specific QIR for a specific target.
/// @param pm Pass Manager to add QIR passes to
/// @param convertTo Expected to be `qir-base` or `qir-adaptive` (comes from the
/// cudaq-translate command line `--convert-to` parameter)
/// \deprecated Replaced by the convert to QIR API pipeline.
void addQIRProfilePipeline(mlir::OpPassManager &pm, llvm::StringRef convertTo);

void addQIRProfileVerify(mlir::OpPassManager &pm, llvm::StringRef convertTo);

void addLowerToCCPipeline(mlir::OpPassManager &pm);
void addWiresetToProfileQIRPipeline(mlir::OpPassManager &pm,
                                    llvm::StringRef profile);

/// Verify that all `CallOp` targets are QIR- or NVQIR-defined functions or in
/// the provided allowed list.
std::unique_ptr<mlir::Pass>
createVerifyNVQIRCallOpsPass(const std::vector<llvm::StringRef> &allowedFuncs);

// Use the addQIRProfilePipeline() for the following passes.
std::unique_ptr<mlir::Pass>
createQIRToQIRProfilePass(llvm::StringRef convertTo);
std::unique_ptr<mlir::Pass> createQIRProfilePreparationPass();
std::unique_ptr<mlir::Pass>
createConvertToQIRFuncPass(llvm::StringRef convertTo);

/// Register target pipelines.
void registerTargetPipelines();

/// Register CodeGenDialect with the provided DialectRegistry.
void registerCodeGenDialect(mlir::DialectRegistry &registry);

mlir::LLVM::LLVMStructType lambdaAsPairOfPointers(mlir::MLIRContext *context);

/// The pipeline for lowering Quake code to the QIR API. There will be three
/// distinct flavors of QIR that can be generated with this pipeline. These
/// are `"qir"`, `"qir-base"`, and `"qir-adaptive"`. This pipeline should be run
/// before conversion to the LLVM-IR dialect.
void registerToQIRAPIPipeline();
void addConvertToQIRAPIPipeline(mlir::OpPassManager &pm, mlir::StringRef api,
                                bool opaquePtr = false);

/// The pipeline for lowering Quake code to the execution manager API. This
/// pipeline should be run before conversion to the LLVM-IR dialect.
void registerToExecutionManagerCCPipeline();

void registerWireSetToProfileQIRPipeline();
void populateCCTypeConversions(mlir::LLVMTypeConverter *converter);

// declarative passes
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"

} // namespace cudaq::opt
