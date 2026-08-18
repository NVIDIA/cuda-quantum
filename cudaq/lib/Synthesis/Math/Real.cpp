/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Math/Real.h"

#include <string>

namespace cudaq::synth {

// 256 bits comfortably absorbs the few-ULP drift that the grid-problem
// machinery is tuned around, without the cost of moving to a wider working
// precision.
mpfr_prec_t Real::default_precision_ = 256;

//===----------------------------------------------------------------------===//
// Real::to_string
//===----------------------------------------------------------------------===//

/// Python-style formatting: fixed-point for exponents roughly in
/// [-4, digits), scientific notation outside that window. The sign is
/// stripped from mpfr_get_str's output up front so the formatting branches
/// operate on an unsigned digit string.
std::string Real::to_string(int32_t digits) const {
  if (mpfr_zero_p(value_))
    return "0.0";

  bool negative = mpfr_sgn(value_) < 0;

  char *str;
  mpfr_exp_t exp;
  str = mpfr_get_str(nullptr, &exp, 10, digits, value_, MPFR_RNDN);

  std::string mantissa = str;
  mpfr_free_str(str);

  // mpfr_get_str prepends '-' for negative inputs; pull it off here so the
  // mantissa is always a positive digit string. The sign is reapplied at
  // the very end.
  if (mantissa[0] == '-')
    mantissa = mantissa.substr(1);

  std::string result;

  int32_t scientific_exp = static_cast<int32_t>(exp) - 1;

  if (scientific_exp >= -4 && scientific_exp < digits) {
    if (exp <= 0) {
      // Pure fraction: "0." + |exp| zeros + mantissa.
      // e.g. exp = -2, mantissa = "1234" -> "0.001234".
      result = "0.";
      for (int32_t i = 0; i < static_cast<int32_t>(-exp); i++)
        result += "0";
      result += mantissa;
    } else if (static_cast<int32_t>(exp) >=
               static_cast<int32_t>(mantissa.length())) {
      // Integer (or near-integer) needing trailing zeros before the dot.
      // e.g. exp = 5, mantissa = "12" -> "12000.0".
      result = mantissa;
      for (int32_t i = static_cast<int32_t>(mantissa.length());
           i < static_cast<int32_t>(exp); i++)
        result += "0";
      result += ".0";
    } else {
      // Mid-range value: insert the decimal point at position `exp`.
      result = mantissa;
      result.insert(static_cast<size_t>(exp), ".");
    }
  } else {
    // Scientific notation: one digit before the dot, the rest after, then a
    // two-digit-zero-padded signed exponent.
    result = mantissa.substr(0, 1);
    if (mantissa.length() > 1) {
      result += ".";
      result += mantissa.substr(1);
    } else {
      result += ".0";
    }

    if (scientific_exp >= 0) {
      result += "e+";
      result += (scientific_exp < 10 ? "0" : "");
      result += std::to_string(scientific_exp);
    } else {
      int32_t abs_exp = -scientific_exp;
      result += "e-";
      result += (abs_exp < 10 ? "0" : "");
      result += std::to_string(abs_exp);
    }
  }

  return negative ? "-" + result : result;
}

} // namespace cudaq::synth
