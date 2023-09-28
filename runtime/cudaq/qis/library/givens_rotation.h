/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <cudaq.h>

namespace cudaq {
struct iswap_pow {
  auto operator()(double exponent, cudaq::qubit &a, cudaq::qubit &b) __qpu__ {
    x<cudaq::ctrl>(a, b);
    h(a);
    x<cudaq::ctrl>(b, a);
    r1(M_PI * exponent / 2.0, a);
    x<cudaq::ctrl>(b, a);
    r1(-M_PI * exponent / 2.0, a);
    h(a);
    x<cudaq::ctrl>(a, b);
  }
};

struct phased_iswap_pow {
  auto operator()(double phase_exponent, double exponent, cudaq::qubit &a,
                  cudaq::qubit &b) __qpu__ {
    r1(M_PI * phase_exponent, a);
    r1(-M_PI * phase_exponent, b);
    iswap_pow{}(exponent, a, b);
    r1(-M_PI * phase_exponent, a);
    r1(M_PI * phase_exponent, b);
  }
};

struct givens_rotation {
  auto operator()(double angle_rads, cudaq::qubit &a, cudaq::qubit &b) __qpu__ {
    const double exponent = 2.0 * angle_rads / M_PI;
    phased_iswap_pow{}(0.25, exponent, a, b);
  }
};
} // namespace cudaq