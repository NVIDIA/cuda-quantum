//===- Passes.h - Fabric transform passes -----------------------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_DIALECT_FABRIC_TRANSFORMS_PASSES_H
#define QLX_DIALECT_FABRIC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace qlx {
namespace fabric {

#define GEN_PASS_DECL
#include "qlx/Dialect/Fabric/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "qlx/Dialect/Fabric/Transforms/Passes.h.inc"

} // namespace fabric
} // namespace qlx

#endif // QLX_DIALECT_FABRIC_TRANSFORMS_PASSES_H
