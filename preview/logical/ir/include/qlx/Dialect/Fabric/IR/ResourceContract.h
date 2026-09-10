/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_FABRIC_IR_RESOURCECONTRACT_H
#define QLX_DIALECT_FABRIC_IR_RESOURCECONTRACT_H

#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"

namespace qlx::fabric {

class PackResourceOp;
class UnpackResourceOp;

using PackResourceVerifier = mlir::LogicalResult (*)(PackResourceOp);
using UnpackResourceVerifier = mlir::LogicalResult (*)(UnpackResourceOp);

/// Register optional semantic proof callbacks for one open resource kind.
/// The Fabric dialect itself validates only generic ownership and schema.
void registerResourceContractVerifier(llvm::StringRef resourceKind,
                                      PackResourceVerifier packVerifier,
                                      UnpackResourceVerifier unpackVerifier);

/// Run a provider proof when the resource kind has registered one. Open
/// resource kinds without a provider retain the generic Fabric contract.
mlir::LogicalResult verifyRegisteredResourcePack(PackResourceOp pack);
mlir::LogicalResult verifyRegisteredResourceUnpack(UnpackResourceOp unpack);

} // namespace qlx::fabric

#endif // QLX_DIALECT_FABRIC_IR_RESOURCECONTRACT_H
