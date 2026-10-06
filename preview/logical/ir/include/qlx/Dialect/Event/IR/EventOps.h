/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_EVENT_EVENTOPS_H
#define QLX_DIALECT_EVENT_EVENTOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "qlx/Dialect/Event/IR/EventDialect.h"
#include "qlx/Dialect/Event/IR/EventInterfaces.h"
#include "qlx/Dialect/Event/IR/EventTypes.h"

#define GET_OP_CLASSES
#include "qlx/Dialect/Event/IR/EventOps.h.inc"

#endif // QLX_DIALECT_EVENT_EVENTOPS_H
