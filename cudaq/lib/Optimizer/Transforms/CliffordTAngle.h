/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Angle classification shared by the passes that snap rotations to Clifford+T
// quarter turns. This file does not define a pass.

#include "llvm/ADT/APFloat.h"
#include <cstdint>
#include <optional>

namespace cudaq::opt::detail {

/// `R1` is a phase gate, so its operator-norm distance uses a different angle
/// scale than the `Rx`, `Ry`, and `Rz` axis rotations.
enum class clifford_t_rotation_kind { axis, r1 };

/// Return mathematical k*pi/denominator rounded to `semantics`. The value is
/// computed in IEEE quad and then rounded to `semantics`. It is the correctly
/// rounded value when `denominator` is a power of two and `semantics` is
/// narrower than quad, and can be off by one unit in the last place otherwise.
/// `denominator` must be positive.
llvm::APFloat canonicalMultipleOfPi(std::int64_t k, std::int64_t denominator,
                                    const llvm::fltSemantics &semantics);

// TODO(https://github.com/NVIDIA/cuda-quantum/issues/5680): Matching within
// roundoff recovers exact angles only after the fact. Carry exact multiples of
// pi through angle arithmetic instead, and have every pass that decides whether
// an angle is exact use one classifier with explicit exact, roundoff, and
// operator-norm modes. Take pi from one source, such as the MPFR constant in
// `cudaq-synth`, instead of the literal here and in other passes.

/// Return k when `angle`, a value a pass computed with an error of at most
/// `roundoff` radians, is within `roundoff` of
/// `canonicalMultipleOfPi(k, denominator)`, and nothing otherwise. The caller
/// replaces the angle with that canonical value. A pass applies this only to an
/// angle it computed; `classifyCliffordTAngle` still matches a constant as
/// written.
std::optional<std::int64_t> matchMultipleOfPi(const llvm::APFloat &angle,
                                              std::int64_t denominator,
                                              double roundoff);

/// Return the floating-point sum of two angles that a pass merges. A sum within
/// rounding error of a multiple of pi/4, zero included, is replaced by that
/// exact multiple. Both angles share one floating-point type.
llvm::APFloat addAngles(const llvm::APFloat &lhs, const llvm::APFloat &rhs);

/// Classify an angle against the nearest k*pi/4 value and return k. Exact mode
/// (`epsilon` of zero) compares the folded constant with the canonical value
/// rounded to the same floating-point semantics. Threshold mode accepts an
/// angle whose rotation is within `epsilon` of the canonical rotation in
/// operator norm. Returns no value when the angle is not a Clifford+T angle.
std::optional<std::int64_t>
classifyCliffordTAngle(const llvm::APFloat &angle, double epsilon,
                       clifford_t_rotation_kind kind);

} // namespace cudaq::opt::detail
