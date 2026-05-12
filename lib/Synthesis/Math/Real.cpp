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

// Static member definition
mpfr_prec_t Real::default_precision_ = 256;

std::string Real::to_string(i32 digits) const {
  if (mpfr_zero_p(value_))
    return "0.0";

  bool negative = mpfr_sgn(value_) < 0;

  // Get the string representation and exponent
  char *str;
  mpfr_exp_t exp;
  str = mpfr_get_str(nullptr, &exp, 10, digits, value_, MPFR_RNDN);

  std::string mantissa = str;
  mpfr_free_str(str);

  // Remove negative sign from mantissa if present (we'll handle it
  // separately)
  if (mantissa[0] == '-')
    mantissa = mantissa.substr(1);

  std::string result;

  // Python-style automatic formatting logic:
  // Use scientific notation if exponent is too large or too small
  // Use regular notation for reasonable ranges
  i32 scientific_exp = static_cast<i32>(exp) - 1;

  if (scientific_exp >= -4 && scientific_exp < digits) {
    // Use regular notation
    if (exp <= 0) {
      // Number is less than 1, e.g., 0.001234
      result = "0.";
      for (i32 i = 0; i < static_cast<i32>(-exp); i++)
        result += "0";
      result += mantissa;
    } else if (static_cast<i32>(exp) >= static_cast<i32>(mantissa.length())) {
      // Number is a whole number or needs trailing zeros
      result = mantissa;
      for (i32 i = static_cast<i32>(mantissa.length());
           i < static_cast<i32>(exp); i++)
        result += "0";
      result += ".0";
    } else {
      // Insert decimal point
      result = mantissa;
      result.insert(static_cast<size_t>(exp), ".");
    }
  } else {
    // Use scientific notation
    result = mantissa.substr(0, 1);
    if (mantissa.length() > 1) {
      result += ".";
      result += mantissa.substr(1);
    } else {
      result += ".0";
    }

    // Add exponent
    if (scientific_exp >= 0) {
      result += "e+";
      result += (scientific_exp < 10 ? "0" : "");
      result += std::to_string(scientific_exp);
    } else {
      i32 abs_exp = -scientific_exp;
      result += "e-";
      result += (abs_exp < 10 ? "0" : "");
      result += std::to_string(abs_exp);
    }
  }

  return negative ? "-" + result : result;
}

} // namespace cudaq::synth
