/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// This header depends on no LLVM or MLIR header, so lowering handlers that
// plan pass pipelines as text can include it.

#include <string_view>

namespace cudaq::opt {

/// The rotations that `clifford-t-optimization-pipeline` merges and snaps and
/// `clifford-t-synthesis-pipeline` replaces with Clifford+T gates, when they
/// are uncontrolled and have constant angles. A lowering that schedules those
/// pipelines keeps these rotations in the basis until synthesis.
inline constexpr std::string_view cliffordTRotations[] = {"r1", "rx", "ry",
                                                          "rz"};

} // namespace cudaq::opt
