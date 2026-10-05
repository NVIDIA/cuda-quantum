/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_PHYS_DIALECT_TD
#define QLX_PHYS_DIALECT_TD

include "mlir/IR/OpBase.td"

def Phys_Dialect : Dialect {
  let name = "phys";
  let cppNamespace = "qlx::phys";
  let summary = "QLX generic physical event graph dialect";
  let description = [{
    The phys dialect owns P3 physical resources, state versions, records,
    partial-order event graphs, architecture bindings, and schedules.
    Hardware-specific dialects may refine these operations without changing
    their ownership and dependency contracts.
  }];
  let useDefaultTypePrinterParser = 1;
}

#endif
