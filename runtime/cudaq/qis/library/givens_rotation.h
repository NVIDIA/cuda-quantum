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
__qpu__ void iswap_pow(double exponent, cudaq::qubit &a, cudaq::qubit &b) {
  x<cudaq::ctrl>(a, b);
  h(a);
  x<cudaq::ctrl>(b, a);
  r1(M_PI * exponent / 2.0, a);
  x<cudaq::ctrl>(b, a);
  r1(-M_PI * exponent / 2.0, a);
  h(a);
  x<cudaq::ctrl>(a, b);
}

template <typename KernelBuilder>
void iswap_pow(KernelBuilder &kernel, cudaq::QuakeValue exponent,
               cudaq::QuakeValue a, cudaq::QuakeValue b) {
  kernel.template x<cudaq::ctrl>(a, b);
  kernel.h(a);
  kernel.template x<cudaq::ctrl>(b, a);
  kernel.r1(M_PI * exponent / 2.0, a);
  kernel.template x<cudaq::ctrl>(b, a);
  kernel.r1(-M_PI * exponent / 2.0, a);
  kernel.h(a);
  kernel.template x<cudaq::ctrl>(a, b);
}

__qpu__ void phased_iswap_pow(double phase_exponent, double exponent,
                              cudaq::qubit &a, cudaq::qubit &b) {
  r1(M_PI * phase_exponent, a);
  r1(-M_PI * phase_exponent, b);
  iswap_pow(exponent, a, b);
  r1(-M_PI * phase_exponent, a);
  r1(M_PI * phase_exponent, b);
}

template <typename KernelBuilder>
void phased_iswap_pow(KernelBuilder &kernel, cudaq::QuakeValue phase_exponent,
                      cudaq::QuakeValue exponent, cudaq::QuakeValue a,
                      const cudaq::QuakeValue b) {
  kernel.r1(M_PI * phase_exponent, a);
  kernel.r1(-M_PI * phase_exponent, b);
  iswap_pow(kernel, exponent, a, b);
  kernel.r1(-M_PI * phase_exponent, a);
  kernel.r1(M_PI * phase_exponent, b);
}

__qpu__ void givens_rotation(double angle_rads, cudaq::qubit &a,
                             cudaq::qubit &b) {
  const double exponent = 2.0 * angle_rads / M_PI;
  phased_iswap_pow(0.25, exponent, a, b);
}

template <typename KernelBuilder>
void givens_rotation(KernelBuilder &kernel, cudaq::QuakeValue angle_rads,
                     cudaq::QuakeValue a, cudaq::QuakeValue b) {
  cudaq::QuakeValue exponent = (2.0 * angle_rads) / M_PI;
  phased_iswap_pow(kernel, kernel.constantVal(0.25), exponent, a, b);
}

template <typename KernelBuilder>
void givens_rotation(KernelBuilder &kernel, double angle_rads,
                     cudaq::QuakeValue a, cudaq::QuakeValue b) {
  givens_rotation(kernel, kernel.constantVal(angle_rads), a, b);
}
} // namespace cudaq