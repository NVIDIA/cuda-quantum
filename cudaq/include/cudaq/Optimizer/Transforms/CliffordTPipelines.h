/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// What a lowering handler needs to know to schedule
// `clifford-t-optimization-pipeline` and `clifford-t-synthesis-pipeline`.
// Handlers plan pipelines as text, so this header includes no LLVM or MLIR
// header. The functions are defined in `OptTransforms`, because their answers
// come from the decomposition-pattern graph.

#include <span>
#include <string>
#include <vector>

namespace cudaq::opt {

/// The basis to decompose to before the Clifford+T pipelines: \p basis plus
/// the rotations those pipelines merge, snap, and synthesize, so that
/// decomposition keeps them for the pipelines to read.
std::vector<std::string>
cliffordTDecompositionBasis(std::span<const std::string> basis);

/// Whether `clifford-t-synthesis-pipeline` approximates rotations for a target
/// with \p basis. It does unless decomposition can lower one of the rotations
/// it reads to the basis exactly, and an empty basis always synthesizes. A
/// stage that approximates needs constant angles, so it needs the kernel's
/// argument values.
bool needsCliffordTSynthesis(std::span<const std::string> basis);

} // namespace cudaq::opt
