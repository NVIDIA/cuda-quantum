/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq::opt {

/// Register the `verify-linear-ownership` pass with the global MLIR pass
/// registry. Out-of-tree tools (such as `qlx-opt`) call this to expose the
/// pass without including Passes.h, whose generated Passes.h.inc is not
/// installed with the SDK. Calling it more than once is harmless.
void registerLinearOwnershipVerifierPass();

} // namespace cudaq::opt
