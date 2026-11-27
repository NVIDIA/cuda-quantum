/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace mlir {
class Block;
}

namespace cudaq::opt {

/// Normalize exact Clifford+T chains in the given block.
void normalizeSingleQubitCliffordT(mlir::Block &block);

} // namespace cudaq::opt
