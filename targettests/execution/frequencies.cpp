/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target qpp-cpu %s -o %t && CUDAQ_LOG_LEVEL=info %t 2>&1 | FileCheck %s
// clang-format on

#include <cassert>
#include <cudaq.h>
#include <iostream>

struct unitary {
  void operator()(cudaq::qarray<2> &q, double theta) __qpu__ {
    ry(theta, q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  }
};

int main(int argc, char **) {
  cudaq::set_random_seed(13);
  auto native = cudaq::sample_vnext(256, unitary{}, 1.5707963267948966);
  assert(native.get_total_shots() == 256 && native.size() == 2);
  assert(native.count("00") > 0 && native.count("11") > 0);

  std::cout << native.count("00") << "\n";
  std::cout << native.count("11") << "\n";
}
