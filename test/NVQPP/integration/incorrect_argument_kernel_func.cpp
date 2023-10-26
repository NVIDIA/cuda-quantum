/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s 2>&1 | FileCheck %s

#include <cudaq.h>

__qpu__ void kernel(double theta) {
  cudaq::qreg q(2);
  x(q[0]);
  ry(theta, q[1]);
  x<cudaq::ctrl>(q[1], q[0]);
}

int main() {

  // Calling sample but not passing along a concrete argument for `value`
  auto result = cudaq::sample(kernel);
}

// CHECK: 'InvalidArgs<cudaq::Msg<{{[0-9]+}}>{"requires <double>, got <>"}>'
