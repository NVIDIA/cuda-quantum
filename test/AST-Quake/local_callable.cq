/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

namespace Ned {
class Carl {
public:
  struct Steve {
    void operator()() __qpu__ {}
  };
};
} // namespace Ned

// CHECK-LABEL: func.func @__nvqpp__mlirgen__N3Ned4Carl5SteveE() attributes

int main() {
  struct applyH {
    auto operator()() __qpu__ {
      cudaq::qubit q;
      h(q);
    }
  };
  auto quake = cudaq::get_quake(applyH{});
  auto quake2 = cudaq::get_quake(Ned::Carl::Steve{});
}

// CHECK-LABEL: func.func @__nvqpp__mlirgen__Z4mainE6applyH() attributes
