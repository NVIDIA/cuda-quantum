/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"

//===----------------------------------------------------------------------===//
//
// Runtime helper functions are functions that will appear in the runtime
// library (implementations are defined in either the headers or libraries in
// the `runtime` directory). These helper functions may never be assumed to
// appear on the device-side, so these helpers should only be used in host-side
// code.
//
//===----------------------------------------------------------------------===//

namespace cudaq::runtime {

/// Get the return type of a kernel FuncOp. Returns a null Type if the kernel
/// returns void or has more than one result (unsupported).
inline mlir::Type getReturnType(mlir::func::FuncOp funcOp) {
  if (!funcOp)
    return nullptr;

  auto numResults = funcOp.getFunctionType().getNumResults();
  return numResults == 1 ? funcOp.getFunctionType().getResult(0) : nullptr;
}

} // namespace cudaq::runtime
