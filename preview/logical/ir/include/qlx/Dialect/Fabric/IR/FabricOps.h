/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef FABRIC_FABRICOPS_H
#define FABRIC_FABRICOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "qlx/Dialect/Event/IR/EventTypes.h"
#include "qlx/Dialect/Fabric/IR/FabricAttrs.h"
#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/Fabric/IR/FabricInterfaces.h"
#include "qlx/Dialect/Fabric/IR/FabricTypes.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"

#define GET_OP_CLASSES
#include "qlx/Dialect/Fabric/IR/FabricOps.h.inc"

namespace qlx::fabric {

/// Return whether two compiler-generated Fabric callables have the same
/// physical implementation modulo their symbol, P1 action-site witness, and
/// selected logical-block provenance.  Those omitted facts are rebound by an
/// explicit P3 resource substitution; every executable operation and all
/// other attributes remain part of the comparison.
bool arePhysicallyTemplateEquivalent(mlir::Operation *lhs,
                                     mlir::Operation *rhs);

} // namespace qlx::fabric

#endif // FABRIC_FABRICOPS_H
