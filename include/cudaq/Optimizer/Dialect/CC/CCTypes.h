/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "llvm/ADT/SmallVector.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h.inc"
