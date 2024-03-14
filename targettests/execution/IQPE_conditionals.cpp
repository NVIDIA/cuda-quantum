/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t

#include <cudaq.h>

struct iqpe {
  void operator()() __qpu__ {
    cudaq::qarray<2> q;
    h(q[0]);
    x(q[1]);
    for (int i = 0; i < 8; i++)
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    h(q[0]);
    auto cr0 = mz(q[0]);
    reset(q[0]);

    h(q[0]);
    for (int i = 0; i < 4; i++)
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    if (cr0)
      rz(-M_PI / 2., q[0]);

    h(q[0]);
    auto cr1 = mz(q[0]);
    reset(q[0]);

    h(q[0]);
    for (int i = 0; i < 2; i++)
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    if (cr0)
      rz(-M_PI / 4., q[0]);

    if (cr1)
      rz(-M_PI / 2., q[0]);

    h(q[0]);
    auto cr2 = mz(q[0]);
    reset(q[0]);
    h(q[0]);
    r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    if (cr0)
      rz(-M_PI / 8., q[0]);

    if (cr1)
      rz(-M_PI_4, q[0]);

    if (cr2)
      rz(-M_PI_2, q[0]);

    h(q[0]);
    mz(q[0]);
  }
};

// CHECK: { 
// CHECK-DAG:   __global__ : { 1:10 }
// CHECK-DAG:   cr0 : { 1:10 }
// CHECK-DAG:   cr1 : { 1:10 }
// CHECK-DAG:   cr2 : { 0:10 }
// CHECK: }

int main() {

  int nShots = 10;
  auto &platform = cudaq::get_platform();
  auto counts = cudaq::sample(nShots, iqpe{});
  counts.dump();

  return 0;
}
