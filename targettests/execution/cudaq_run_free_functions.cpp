/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>

struct foo_test_result {
  int64_t a;
  int64_t b;
};

foo_test_result foo_test(int x) __qpu__ {
  cudaq::qvector q(x);
  h(q[0]);
  for (int i = 1; i < x; i++) {
    cx(q[i - 1], q[i]);
  }
  return {9, 10};
}

struct bar_test_result {
  bool a;
};

bar_test_result bar_test() __qpu__ {
  return {true};
}

struct baz_test_result {
  bool a;
  int b;
  int64_t c;
};

baz_test_result baz_test() __qpu__ {
  cudaq::qvector q(5);
  h(q[0]);
  for (int i = 1; i < 5; i++) {
    cx(q[i - 1], q[i]);
  }
  return baz_test_result{true, 9, 10l};
};

int main() {
  {
    const auto results = cudaq::run(5, foo_test, 2 + 3);
    for (auto i : results)
      printf("%ld, %ld\t", i.a, i.b);
    printf("\n");
  }
  {
    const auto results = cudaq::run(4, bar_test);
    for (auto i : results)
      printf("%d\t", (int)i.a);
    printf("\n");
  }
  {
    const auto results = cudaq::run(3, baz_test);
    for (auto i : results)
      printf("%d, %d, %ld\t", (int)i.a, i.b, i.c);
    printf("\n");
  }
  return 0;
}

// CHECK: 9, 10   9, 10   9, 10   9, 10   9, 10
// CHECK: 1       1       1       1
// CHECK: 1, 9, 10        1, 9, 10        1, 9, 10
