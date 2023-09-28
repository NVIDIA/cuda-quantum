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
struct fermionic_swap {
  auto operator()(double phi, cudaq::qubit &a, cudaq::qubit &b) __qpu__ {
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
};

} // namespace cudaq