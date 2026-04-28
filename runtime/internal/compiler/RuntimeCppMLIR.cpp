/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "cudaq_internal/compiler/TracePassInstrumentation.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

void cudaq_internal::compiler::initializeLangMLIR() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  cudaq::registerAllPasses();
}

mlir::LogicalResult
cudaq_internal::compiler::runPassManager(mlir::PassManager &pm,
                                         mlir::Operation *op) {
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  return pm.run(op);
}
