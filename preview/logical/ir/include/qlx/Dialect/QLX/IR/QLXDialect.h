//===- QLXDialect.h - QLX dialect declaration ------------------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_DIALECT_QLX_QLXDIALECT_H
#define QLX_DIALECT_QLX_QLXDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"

#include <optional>

namespace qlx {

/// Typed cross-dialect queries needed by the qlx.device verifier. Implemented
/// by LVM and Fabric without making the QLX dialect depend on their concrete
/// attribute or type classes.
class DeviceBindingDialectInterface
    : public mlir::DialectInterface::Base<DeviceBindingDialectInterface> {
public:
  using Base::Base;

  virtual std::optional<llvm::StringRef>
  getLogicalCapabilityKey(mlir::Attribute) const {
    return std::nullopt;
  }
  virtual std::optional<llvm::StringRef>
  getLogicalServiceKey(mlir::Attribute) const {
    return std::nullopt;
  }
  virtual mlir::FlatSymbolRefAttr getResourceKind(mlir::Type) const {
    return {};
  }
};

} // namespace qlx

MLIR_DECLARE_EXPLICIT_TYPE_ID(qlx::DeviceBindingDialectInterface)

#include "qlx/Dialect/QLX/IR/QLXDialect.h.inc"

#endif // QLX_DIALECT_QLX_QLXDIALECT_H
