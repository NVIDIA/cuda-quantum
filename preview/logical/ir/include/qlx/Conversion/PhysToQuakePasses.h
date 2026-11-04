/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Declares `phys-to-quake`, which lowers a P3 physical graph to Quake. It
// needs CUDA-Q core's Quake and CC dialects, so it is built only when
// `QLX_USE_CUDAQ_SDK` is ON, the default.

#include "mlir/Pass/Pass.h"

namespace qlx {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "qlx/Conversion/PhysToQuakePasses.h.inc"

} // namespace qlx
