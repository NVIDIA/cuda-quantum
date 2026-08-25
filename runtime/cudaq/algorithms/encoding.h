/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include "cudaq/qis/state.h"
#include <complex>
#include <span>
#include <vector>

namespace cudaq {
namespace contrib {

/// @brief Map classical features to a normalized quantum state by amplitude
/// encoding.
///
/// Amplitude encoding represents a classical feature vector as the amplitudes
/// of a pure state in the computational basis. Given a length-\f$d\f$ vector
/// \f$\mathbf{x} = (x_0, \ldots, x_{d-1})\f$ (real or complex), the procedure
/// is:
///
/// 1. **Pad** to length \f$N = 2^n\f$ for the smallest \f$n\f$ with \f$2^n \ge
/// d\f$.
///    The padded vector \f$\mathbf{x}'\f$ satisfies \f$x'_i = x_i\f$ for
///    \f$i < d\f$ and \f$x'_i = \texttt{pad}\f$ for \f$d \le i < N\f$.
///
/// 2. **Normalize** with the Euclidean (L2) norm (must be non-zero).
///    Coefficients are \f$α_i = x'_i / \|\mathbf{x}'\|_2\f$.
///
/// 3. **Form the state** in the \f$n\f$-qubit computational basis:
///    \f$|\psi\rangle = \sum_{i=0}^{N-1} α_i |i\rangle\f$, where
///    \f$|i\rangle\f$ is the basis ``ket`` with index \f$i\f$ in binary.
///
/// The returned ``cudaq::state`` stores \f$α_i\f$ in little-endian index
/// order (consistent with ``qvector(state)``). Real inputs are promoted to
/// complex amplitudes with zero imaginary part before padding.
///
/// @param data Classical features as a 1D real or complex vector.
/// @param pad Value used when padding to the nearest ``2^n`` (default ``0``).
/// @throws std::invalid_argument if ``data`` is empty or has zero norm after
/// padding.
state amplitude_encode(std::span<const double> data,
                       std::complex<double> pad = 0.0);

/// @copydoc amplitude_encode
state amplitude_encode(std::span<const float> data,
                       std::complex<double> pad = 0.0);

/// @copydoc amplitude_encode
state amplitude_encode(std::span<const std::complex<double>> data,
                       std::complex<double> pad = 0.0);

/// @copydoc amplitude_encode
state amplitude_encode(std::span<const std::complex<float>> data,
                       std::complex<double> pad = 0.0);

/// @copydoc amplitude_encode
state amplitude_encode(const state &data, std::complex<double> pad = 0.0);

/// @copydoc amplitude_encode
template <typename T>
state amplitude_encode(const std::vector<T> &data,
                       std::complex<double> pad = 0.0) {
  return amplitude_encode(std::span<const T>(data), pad);
}

} // namespace contrib
} // namespace cudaq
