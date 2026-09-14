/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx-c/Dialect/Event.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

#include "qlx/Dialect/Event/IR/EventDialect.h"
#include "qlx/Dialect/Event/IR/EventTypes.h"

#include "mlir/IR/BuiltinAttributes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Event, event, qlx::event::EventDialect)

//===----------------------------------------------------------------------===//
// !event.handle<payload, ownership, stream?>
//===----------------------------------------------------------------------===//

bool eventTypeIsAHandle(MlirType type) {
  return llvm::isa<qlx::event::HandleType>(unwrap(type));
}

MlirType eventHandleTypeGet(MlirContext ctx, MlirType payload,
                            MlirStringRef ownership, MlirAttribute stream) {
  mlir::SymbolRefAttr streamAttr;
  if (!mlirAttributeIsNull(stream)) {
    streamAttr = llvm::dyn_cast<mlir::SymbolRefAttr>(unwrap(stream));
    if (!streamAttr)
      return MlirType{nullptr};
  }
  return wrap(qlx::event::HandleType::get(unwrap(ctx), unwrap(payload),
                                          unwrap(ownership), streamAttr));
}

MlirType eventHandleTypeGetPayload(MlirType type) {
  return wrap(llvm::cast<qlx::event::HandleType>(unwrap(type)).getPayload());
}

MlirStringRef eventHandleTypeGetOwnership(MlirType type) {
  return wrap(llvm::cast<qlx::event::HandleType>(unwrap(type)).getOwnership());
}

MlirAttribute eventHandleTypeGetStream(MlirType type) {
  mlir::SymbolRefAttr stream =
      llvm::cast<qlx::event::HandleType>(unwrap(type)).getStream();
  if (!stream)
    return MlirAttribute{nullptr};
  return wrap(static_cast<mlir::Attribute>(stream));
}

MlirTypeID eventHandleTypeGetTypeID(void) {
  return wrap(qlx::event::HandleType::getTypeID());
}
