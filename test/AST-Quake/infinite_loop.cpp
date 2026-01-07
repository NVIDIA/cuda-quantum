/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --canonicalize --cc-loop-normalize |& FileCheck %s

#include <cudaq.h>

// Counted loop structure when condition is always true

__qpu__ int t1() {
  cudaq::qubit q;
  for (std::uint32_t u = 1; u <= 0xffffffff; u++)
    x(q);
  return 0;
}

__qpu__ int t2() {
  cudaq::qubit q;
  for (std::int32_t u = 1; u <= 0x7fffffff; u++)
    x(q);
  return 0;
}

__qpu__ int t3() {
  cudaq::qubit q;
  for (std::uint64_t u = 5; u <= 0xffffffffffffffff; u++)
    x(q);
  return 0;
}

__qpu__ int t4() {
  cudaq::qubit q;
  for (std::int64_t u = 16; u <= 0x7fffffffffffffff; u++)
    x(q);
  return 0;
}

__qpu__ int t5() {
  cudaq::qubit q;
  for (std::uint64_t u = -14; u >= 0; u--)
    x(q);
  return 0;
}

__qpu__ int t6() {
  cudaq::qubit q;
  std::int64_t cmp = 0x8000000000000000;
  for (std::int64_t u = 83; u >= cmp; u++)
    x(q);
  return 0;
}

// CHECK: Loop condition is always true. This loop is not supported in a kernel.
// CHECK: Loop condition is always true. This loop is not supported in a kernel.
// CHECK: Loop condition is always true. This loop is not supported in a kernel.
// CHECK: Loop condition is always true. This loop is not supported in a kernel.
// CHECK: Loop condition is always true. This loop is not supported in a kernel.
// CHECK: Loop condition is always true. This loop is not supported in a kernel.
