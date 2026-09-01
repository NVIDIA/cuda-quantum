//===- FabricOps.h - Fabric operation declarations ---------------*- C++
//-*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef FABRIC_FABRICOPS_H
#define FABRIC_FABRICOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "qlx/Dialect/Fabric/IR/FabricAttrs.h"
#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/Fabric/IR/FabricInterfaces.h"
#include "qlx/Dialect/Fabric/IR/FabricTypes.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"

#define GET_OP_CLASSES
#include "qlx/Dialect/Fabric/IR/FabricOps.h.inc"

#endif // FABRIC_FABRICOPS_H
