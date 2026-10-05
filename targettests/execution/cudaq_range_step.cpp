/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s

// Regression test for `cudaq::range(start, stop, step)`:
//  - the runtime intrinsic `__nvqpp_CudaqSizeFromTriple` (which computes how
//    many iterations the triple describes) had its branch targets swapped,
//    so it always returned 0 for a non-zero step and divided by zero for a
//    zero step;
//  - `for (auto i : cudaq::range(start, stop, step))` bound the loop
//    variable to the loop's 0-based iteration counter instead of the actual
//    stepped value.
// Both bugs silently produced wrong results (or 0) for any count-up,
// count-down, break, or continue over a stepped range.

#include <cudaq.h>

__qpu__ std::int64_t count_up() {
  std::int64_t total = 0;
  for (auto i : cudaq::range(2, 20, 3)) {
    total += i;
  }
  return total;
}

__qpu__ std::int64_t count_down() {
  std::int64_t total = 0;
  for (auto i : cudaq::range(10, 0, -2)) {
    total += i;
  }
  return total;
}

__qpu__ std::int64_t empty_range() {
  std::int64_t total = 0;
  for (auto i : cudaq::range(5, 3, 1)) {
    total += 1;
  }
  return total;
}

__qpu__ std::int64_t stepped_break(std::int64_t n) {
  std::int64_t total = 0;
  for (auto i : cudaq::range(2, 20, 3)) {
    if (i == n)
      break;
    total += i;
  }
  return total;
}

__qpu__ std::int64_t stepped_continue(std::int64_t n) {
  std::int64_t total = 0;
  for (auto i : cudaq::range(2, 20, 3)) {
    if (i == n)
      continue;
    total += i;
  }
  return total;
}

int main() {
  printf("count_up: %lld\n", (long long)count_up());
  printf("count_down: %lld\n", (long long)count_down());
  printf("empty_range: %lld\n", (long long)empty_range());
  printf("stepped_break: %lld\n", (long long)stepped_break(8));
  printf("stepped_continue: %lld\n", (long long)stepped_continue(8));
  return 0;
}

// CHECK: count_up: 57
// CHECK: count_down: 30
// CHECK: empty_range: 0
// CHECK: stepped_break: 7
// CHECK: stepped_continue: 49
