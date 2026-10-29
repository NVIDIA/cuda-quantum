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

#include <string>
#include <string_view>
#include <vector>

namespace cudaq::opt {

/// The rotations that `clifford-t-optimization-pipeline` merges and snaps and
/// `clifford-t-synthesis-pipeline` replaces with Clifford+T gates, when they
/// are uncontrolled and have constant angles. A lowering that schedules those
/// pipelines keeps these rotations in the basis until synthesis.
inline constexpr std::string_view cliffordTRotations[] = {"r1", "rx", "ry",
                                                          "rz"};

/// Whether `clifford-t-synthesis-pipeline` approximates rotations for a target
/// with \p basis. It does unless decomposition can lower one of
/// `cliffordTRotations` to the basis exactly. An empty basis always
/// synthesizes. Lowering handlers use this to tell whether a stage will need
/// constant angles. It is defined in `OptTransforms`, because the answer comes
/// from the decomposition-pattern graph, so a caller links that library.
bool needsCliffordTSynthesis(const std::vector<std::string> &basis);

} // namespace cudaq::opt
