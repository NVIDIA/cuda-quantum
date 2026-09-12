/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_PHYS_IR_SPACETIMEDERIVATION_H
#define QLX_DIALECT_PHYS_IR_SPACETIMEDERIVATION_H

#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace qlx::phys {

class SpacetimePlanOp;
class SpacetimeCallOp;

/// Provider callback that independently authenticates one derived P3 plan.
///
/// The Phys dialect owns only the provider-neutral plan representation and
/// registry. Algorithm and paper specific providers register their verifier
/// from the compiler component that implements the derivation.
using SpacetimePlanVerifier = mlir::LogicalResult (*)(SpacetimePlanOp plan);
using SpacetimeCallVerifier = mlir::LogicalResult (*)(SpacetimeCallOp call,
                                                      SpacetimePlanOp plan);

/// Register one exact provider/derivation contract. Registration is
/// idempotent for the same callback and fails hard on conflicting ownership.
void registerSpacetimePlanVerifier(
    llvm::StringRef provider, llvm::StringRef providerVersion,
    llvm::StringRef derivation, std::int64_t derivationVersion,
    SpacetimePlanVerifier verifier,
    SpacetimeCallVerifier callVerifier = nullptr);

/// Verify a plan through its registered provider. Unknown tuples fail closed.
mlir::LogicalResult verifyRegisteredSpacetimeDerivation(SpacetimePlanOp plan);

/// Authenticate an invocation that may reuse a provider plan derived from a
/// structurally equivalent retained P2 protocol. Providers that do not opt in
/// remain exact-source-only.
mlir::LogicalResult verifyRegisteredSpacetimeInvocation(SpacetimeCallOp call,
                                                        SpacetimePlanOp plan);

} // namespace qlx::phys

#endif // QLX_DIALECT_PHYS_IR_SPACETIMEDERIVATION_H
