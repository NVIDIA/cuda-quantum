/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef TEST_FTQC_OPS_H
#define TEST_FTQC_OPS_H

#include "CUDAQLogical/Interfaces/PhysicalInterfaces.h"
#include "test_ftqc/TestFTQCDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "test_ftqc/TestFTQCOps.h.inc"

#endif // TEST_FTQC_OPS_H
