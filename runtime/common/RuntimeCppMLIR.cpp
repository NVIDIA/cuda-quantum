/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/InitAllPasses.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"

namespace cudaq {

void initializeLangMLIR() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  cudaq::registerAllPasses();
}
} // namespace cudaq
