/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Ring/Domega.h"
#include "cudaq/Synthesis/Math/Real.h"

#include <array>
#include <optional>
#include <utility>

namespace cudaq::synth {

/// Abstract interface for convex subsets of ℝ² used by the grid/ellipse
/// algorithms.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Remark 5.4 and §5.
/// The paper assumes that each convex set A is “effectively given” by simple
/// geometric oracles. In particular, the algorithms need to be able to:
///
///   1. Decide membership: given a point v, is v ∈ A?
///   2. Intersect a line (or ray) with A: given a `parametrized` line u(t),
///      find the interval of t for which u(t) lies in A.
///
/// This base class exposes exactly these operations in 2D. Concrete subclasses
/// (unit disc, rectangle, ε-region, ellipse, etc.) implement the details of:
///
///   - contains(): membership oracle v ↦ [v ∈ A?]
///   - intersect(): line / ray intersection oracle
///
/// so that the two-dimensional grid problem (TDGP) solver can treat all such
/// sets uniformly, as required by Theorem 5.18.
class ConvexSet {
public:
  virtual ~ConvexSet() = default;

  /// Membership oracle:
  ///   contains(v) == true <=> v lies in the convex set (up to numerical
  ///   tolerance).
  virtual bool contains(const DOmega &v) const = 0;

  /// Line / ray intersection oracle.
  ///
  /// The caller `parameterizes` a line (or ray) as
  ///   u(t) = u0 + t·v,  t ∈ ℝ,
  /// and intersect() returns the interval [t_min, t_max] for which u(t) lies
  /// inside the convex set, or std::nullopt if there is no intersection at all.
  ///
  /// Concrete implementations are free to interpret [t_min, t_max] as:
  ///   - a full line segment intersection, or
  ///   - a restricted ray (e.g. t ≥ 0) intersection,
  /// depending on how the caller uses t.
  virtual std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0, const DOmega &v) const = 0;
};

} // namespace cudaq::synth
