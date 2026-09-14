/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_PHYS_PHYSOPS_H
#define QLX_DIALECT_PHYS_PHYSOPS_H
#include "qlx/Dialect/Event/IR/EventTypes.h"
#include "qlx/Dialect/Phys/IR/PhysDialect.h"
#include "qlx/Dialect/Phys/IR/PhysTypes.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#define GET_OP_CLASSES
#include "qlx/Dialect/Phys/IR/PhysOps.h.inc"
#endif
