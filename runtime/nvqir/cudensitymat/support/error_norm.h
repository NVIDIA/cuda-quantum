/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

/// \file error_norm.h
/// \brief Device-side scaled error norm for the adaptive (dopri5) integrator.
///
/// The reduction runs entirely on the GPU so the adaptive controller does not
/// copy the two trial states back to host on every step. Only a single scalar
/// is returned to the host.

#include <cstddef>

namespace cudaq::detail {

/// \brief Accumulate the local scipy-style scaled-error sum of squares between
///        two candidate solutions.
///
/// Computes, on the device,
///   sum_i ( |y5_i - y4_i| / (atol + rtol * max(|y5_i|, |y4_i|)) )^2
/// over the \p n complex elements of the local shard and writes the result to
/// \p sumsq_out (host). The caller is responsible for combining the partial
/// sums (and element counts) across MPI ranks and forming the RMS norm
///   sqrt( sum_sumsq / sum_n ).
///
/// \param y5 Device pointer to the fifth-order solution (cuDoubleComplex[n]).
/// \param y4 Device pointer to the embedded fourth-order solution.
/// \param n Number of complex elements in the local shard.
/// \param rtol Relative tolerance.
/// \param atol Absolute tolerance.
/// \param sumsq_out Host output: local sum of squared scaled errors.
void scaled_error_sumsq(const void *y5, const void *y4, std::size_t n,
                        double rtol, double atol, double *sumsq_out);

} // namespace cudaq::detail
