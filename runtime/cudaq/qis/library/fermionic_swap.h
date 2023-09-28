/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/builder/kernel_builder.h"
#include <cudaq.h>
namespace cudaq {
__qpu__ void fermionic_swap(double phi, cudaq::qubit &a, cudaq::qubit &b) {
  h(a);
  h(b);

  x<cudaq::ctrl>(a, b);
  rz(phi / 2.0, b);
  x<cudaq::ctrl>(a, b);

  h(a);
  h(b);

  rx(M_PI_2, a);
  rx(M_PI_2, b);

  x<cudaq::ctrl>(a, b);
  rz(phi / 2.0, b);
  x<cudaq::ctrl>(a, b);

  rx(-M_PI_2, a);
  rx(-M_PI_2, b);
  rz(phi / 2.0, a);
  rz(phi / 2.0, b);
}

template <typename KernelBuilder>
void fermionic_swap(KernelBuilder &kernel, cudaq::QuakeValue &phi,
                    cudaq::QuakeValue &a, cudaq::QuakeValue &b) {
  kernel.h(a);
  kernel.h(b);

  kernel.template x<cudaq::ctrl>(a, b);
  kernel.rz(phi / 2.0, b);
  kernel.template x<cudaq::ctrl>(a, b);

  kernel.h(a);
  kernel.h(b);

  kernel.rx(M_PI_2, a);
  kernel.rx(M_PI_2, b);

  kernel.template x<cudaq::ctrl>(a, b);
  kernel.rz(phi / 2.0, b);
  kernel.template x<cudaq::ctrl>(a, b);

  kernel.rx(-M_PI_2, a);
  kernel.rx(-M_PI_2, b);
  kernel.rz(phi / 2.0, a);
  kernel.rz(phi / 2.0, b);
}
} // namespace cudaq