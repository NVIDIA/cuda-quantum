/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>

/// Canonical fixed-width integer aliases for the synthesizer.
///
/// All synthesizer code should use these aliases instead of `int`, `long`,
/// or `long long`. The only exceptions are:
///
///   - `size_t` for STL sizes and array indexing (C++ convention).
///   - `long` / `unsigned long` in function bodies that directly call GMP/MPFR
///     APIs (e.g. mpz_set_si, mpz_add_ui), where the library ABI mandates
///     those types. At such callsites use an explicit static_cast<long> /
///     static_cast<unsigned long> to document the boundary.
///   - GMP/MPFR-semantic types (mp_bitcnt_t, mpfr_prec_t, gmp_randstate_t)
///     which carry domain meaning beyond their width.

namespace cudaq::synth {

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

} // namespace cudaq::synth
