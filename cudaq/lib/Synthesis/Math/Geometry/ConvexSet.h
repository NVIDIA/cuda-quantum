/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Ring/Domega.h"

#include <array>
#include <optional>
#include <utility>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// ConvexSet
//===----------------------------------------------------------------------===//

/// Abstract interface for convex subsets of R^2 used by the grid / ellipse
/// algorithms.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Remark 5.4 and sec. 5. The
/// paper assumes each convex set is "effectively given" by two geometric
/// oracles; this base class exposes exactly those two operations so that the
/// TDGP solver (Theorem 5.18) can treat unit disk, rectangle, epsilon-region,
/// ellipse, etc. uniformly.
class ConvexSet {
public:
  virtual ~ConvexSet() = default;

  /// Membership oracle: true iff v lies in the set (modulo numerical
  /// tolerance defined by concrete subclasses).
  virtual bool contains(const DOmega &v) const = 0;

  /// Line / ray intersection oracle.
  ///
  /// The caller parameterizes a line as u(t) = u0 + t*v with t in R. The
  /// return value is the interval [t_min, t_max] for which u(t) lies inside
  /// the set, or std::nullopt if there is no intersection. Concrete
  /// implementations are free to interpret the result as a full line
  /// intersection or as a restricted (e.g. t >= 0) ray, depending on what
  /// the caller does with t.
  virtual std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0, const DOmega &v) const = 0;
};

} // namespace cudaq::synth
