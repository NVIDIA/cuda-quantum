/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/InliningUtils.h"

namespace cudaq {

/// Generic opt-in with inlining any operation. Inlining kernels is important to
/// build up larger functions to translate into quantum circuits.
/// Do not allow kernel entry points to be inlined into their callers, typically
/// the thunk function.
struct EnableInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Region *, mlir::Region *src, bool,
                       mlir::IRMapping &) const final {
    if (auto func = src->getParentOfType<mlir::func::FuncOp>())
      return !(func->hasAttr(cudaq::entryPointAttrName));
    return true;
  }
  bool isLegalToInline(mlir::Operation *op, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    if (auto func = op->getParentOfType<mlir::func::FuncOp>())
      return !(func->hasAttr(cudaq::entryPointAttrName));
    return true;
  }
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool) const final;
};

} // namespace cudaq
