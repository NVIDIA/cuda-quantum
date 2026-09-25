/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_PHYS_PHYSDIALECT_H
#define QLX_DIALECT_PHYS_PHYSDIALECT_H
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "qlx/Dialect/Phys/IR/PhysDialect.h.inc"

namespace qlx::phys {

/// Close every deferred graph/profile/projection link carried by a detached
/// QEC sidecar for `graphSymbol`.  Generic dialect verification intentionally
/// permits unresolved symbols in partial libraries; terminal consumers call
/// this helper after selecting a concrete linked graph.
mlir::LogicalResult verifyClosedSidecarLinks(mlir::ModuleOp module,
                                             llvm::StringRef graphSymbol);

/// Fully verify a set of detached QEC sidecars for one physical graph while
/// sharing the immutable graph record/call index across their ordinary ODS
/// verifiers.  This changes verifier work, never the accepted IR contract.
mlir::LogicalResult
verifySidecarBatch(llvm::ArrayRef<mlir::Operation *> sidecars);

} // namespace qlx::phys
#endif
