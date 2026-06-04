/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.  * All rights reserved.
 *                                                      *
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

/// @brief Map classical features to a normalized quantum state by amplitude
/// encoding.
///
/// Pads ``data`` to the next power of two (using ``pad``), L2-normalizes, and
/// returns a :class:`cudaq::state` suitable for simulation and
/// ``qvector(state)`` initialization.
///
/// @param data Classical features as a 1D real or complex vector.
/// @param pad Value used when padding to the nearest ``2^n`` (default ``0``).
state amplitude_encode(std::span<const double> data,
                       std::complex<double> pad = 0.0);

/// @brief Amplitude-encode a single-precision real vector.
state amplitude_encode(std::span<const float> data,
                       std::complex<double> pad = 0.0);

/// @brief Amplitude-encode a double-precision complex vector.
state amplitude_encode(std::span<const std::complex<double>> data,
                       std::complex<double> pad = 0.0);

/// @brief Amplitude-encode a single-precision complex vector.
state amplitude_encode(std::span<const std::complex<float>> data,
                       std::complex<double> pad = 0.0);

/// @brief Amplitude-encode from an existing state vector.
state amplitude_encode(const state &data, std::complex<double> pad = 0.0);

/// @brief Amplitude-encode a ``std::vector`` of classical features.
template <typename T>
state amplitude_encode(const std::vector<T> &data,
                       std::complex<double> pad = 0.0) {
  return amplitude_encode(std::span<const T>(data), pad);
}

} // namespace cudaq
