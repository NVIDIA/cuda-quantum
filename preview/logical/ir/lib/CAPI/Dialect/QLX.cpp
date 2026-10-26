/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx-c/Dialect/QLX.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"
#include "qlx/Dialect/QLX/Transforms/QLXToPBC.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(QLX, qlx, qlx::QLXDialect)

bool qlxLowerToPBC(MlirModule module) {
  return mlir::succeeded(qlx::lowerToPBC(unwrap(module)));
}

//===----------------------------------------------------------------------===//
// Enum attributes
//===----------------------------------------------------------------------===//

bool qlxAttributeIsAPauli(MlirAttribute attr) {
  return llvm::isa<qlx::PauliAttr>(unwrap(attr));
}

MlirAttribute qlxPauliAttrGet(MlirContext ctx, MlirStringRef value) {
  auto sym = qlx::symbolizePauli(unwrap(value));
  if (!sym)
    return MlirAttribute{nullptr};
  return wrap(qlx::PauliAttr::get(unwrap(ctx), *sym));
}

MlirStringRef qlxPauliAttrGetValue(MlirAttribute attr) {
  auto a = llvm::cast<qlx::PauliAttr>(unwrap(attr));
  return wrap(qlx::stringifyPauli(a.getValue()));
}

MlirTypeID qlxPauliAttrGetTypeID(void) {
  return wrap(qlx::PauliAttr::getTypeID());
}
