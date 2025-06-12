/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ =fenable-cudaq-run %s -o %t && %t | FileCheck %s

#include <cudaq.h>

struct MyTuple {
  bool boolVal;
  std::int64_t i64Val;
  double f64Val;
};

auto struct_test = [](MyTuple t) __qpu__ { return t; };

int main() {
  int c = 0;
  {
    MyTuple t = {true, 654, 9.123};
    const auto results = cudaq::run(3, struct_test, t);
    if (results.size() != 3) {
      printf("FAILED! Expected 3 shots. Got %lu\n", results.size());
    } else {
      c = 0;
      for (auto i : results)
        printf("%d: {%s, %ld, %f}\n", c++, i.boolVal ? "true" : "false",
               i.i64Val, i.f64Val);
      printf("success!\n");
    }
  }
  return 0;
}

// CHECK: 0: {true, 654, 9.123000}
// CHECK: 1: {true, 654, 9.123000}
// CHECK: 2: {true, 654, 9.123000}
// CHECK: success!
