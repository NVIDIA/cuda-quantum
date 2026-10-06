/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/APFloat.h"
#include <cstdint>
#include <optional>

namespace cudaq::opt::detail {

/// `R1` is a phase gate, so its operator-norm distance uses a different angle
/// scale than the `Rx`, `Ry`, and `Rz` axis rotations.
enum class clifford_t_rotation_kind { axis, r1 };

/// Reconstruct the correctly rounded value of mathematical k*pi/4 in
/// `semantics`.
llvm::APFloat canonicalQuarterTurns(int64_t quarterTurns,
                                    const llvm::fltSemantics &semantics);

/// Classify an angle against the nearest k*pi/4 value and return k. Exact mode
/// (`epsilon` of zero) compares the folded constant with the canonical value
/// rounded to the same floating-point semantics. Threshold mode accepts an
/// angle whose rotation is within `epsilon` of the canonical rotation in
/// operator norm. Returns no value when the angle is not a Clifford+T angle.
std::optional<int64_t> classifyCliffordTAngle(const llvm::APFloat &angle,
                                              double epsilon,
                                              clifford_t_rotation_kind kind);

} // namespace cudaq::opt::detail
