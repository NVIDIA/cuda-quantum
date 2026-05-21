/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Fixed-width integer aliases
//===----------------------------------------------------------------------===//
//
// Synthesis code uses these aliases instead of `int`, `long`, or `long long`.
// Exceptions:
//   - `size_t` for STL sizes and array indexing (C++ convention).
//   - `long` / `unsigned long` in function bodies that call GMP/MPFR APIs
//     directly (e.g. mpz_set_si). At those boundaries an explicit
//     static_cast<long> or static_cast<unsigned long> documents the
//     library-ABI conversion.
//   - GMP/MPFR-semantic types (mp_bitcnt_t, mpfr_prec_t, gmp_randstate_t),
//     which carry domain meaning beyond their bit width.

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

} // namespace cudaq::synth
