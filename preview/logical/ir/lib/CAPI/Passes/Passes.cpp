//===- Passes.cpp - QLX pass registration C-API impl --------------------===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#include "qlx-c/Passes.h"

#include "qlx/Dialect/QLX/Transforms/QLXSynthesize.h"
#include "qlx/Dialect/QLX/Transforms/QLXToPBC.h"
#include "qlx/Dialect/QLX/Transforms/QLXVerifyPBC.h"
#include "qlx/InitAllPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"

#include <string>

void qlxRegisterAllPasses(void) { qlx::registerAllQLXPasses(); }

MlirLogicalResult qlxVerifyCliffordTModule(MlirModule module) {
  return mlir::succeeded(qlx::verifyCliffordT(unwrap(module)))
             ? mlirLogicalResultSuccess()
             : mlirLogicalResultFailure();
}

MlirLogicalResult qlxLowerToPBCModule(MlirModule module) {
  return mlir::succeeded(qlx::lowerToPBC(unwrap(module)))
             ? mlirLogicalResultSuccess()
             : mlirLogicalResultFailure();
}

MlirLogicalResult qlxVerifyPBCModule(MlirModule module) {
  std::string error;
  return mlir::succeeded(qlx::verifyPBCForm(unwrap(module), error))
             ? mlirLogicalResultSuccess()
             : mlirLogicalResultFailure();
}
