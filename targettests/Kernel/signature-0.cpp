/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iomanip>
#include <iostream>

// Test kernels can take arguments of void, arithmetic, and std::vector<bool> as
// well as return values of same.

void ok() { std::cout << "ok\n"; }

void fail() { std::cout << "fail\n"; }

struct Qernel1 {
  void operator()() __qpu__ { ok(); }
};

struct Qernel2 {
  void operator()(double d) __qpu__ {
    if (d == 12.0)
      ok();
    else
      fail();
  }
};

struct Qernel3 {
  int operator()(double d) __qpu__ {
    if (d == 13.1)
      ok();
    else
      fail();
    return 14;
  }
};

struct Qernel4 {
  float operator()() __qpu__ {
    ok();
    return 15.2f;
  }
};

class Qernel5 {
public:
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(5);
    return mz(q);
  }
};

class Qernel6 {
public:
  std::vector<bool> operator()(int sz) __qpu__ {
    cudaq::qvector q(sz);
    return mz(q);
  }
};

// FIXME: unhandled ctor call
#define NYI /*__qpu__*/

class Qernel7 {
public:
  std::vector<bool> operator()(std::vector<bool> v) NYI { return v; }
};

int main() {
  std::cout << "Qernel1 ";
  Qernel1{}();

  std::cout << "Qernel2 ";
  Qernel2{}(12.0);

  std::cout << "Qernel3 ";
  auto r1 = Qernel3{}(13.1);
  if (r1 == 14)
    ok();

  std::cout << "Qernel4 ";
  auto r2 = Qernel4{}();
  if (r2 == 15.2f)
    ok();

  std::cout << "Qernel5 ";
  auto r3 = Qernel5{}();
  if (r3.size() == 5)
    ok();

  std::cout << "Qernel6 ";
  auto r4 = Qernel6{}(3);
  if (r4.size() == 3)
    ok();

  std::cout << "Qernel7 ";
  std::vector<bool> in1 = {true, false, true, true};
  auto r5 = Qernel7{}(in1);
  if (r5.size() == 4 && r5[0] && !r5[1] && r5[2] && r5[3])
    ok();

  return 0;
}

// clang-format off
// CHECK-LABEL: Qernel1 ok
// CHECK-NEXT: Qernel2 ok
// CHECK-NEXT: Qernel3 ok
// CHECK-NEXT: ok
// CHECK-NEXT: Qernel4 ok
// CHECK-NEXT: ok
// CHECK-NEXT: Qernel5 ok
// CHECK-NEXT: Qernel6 ok
// CHECK-NEXT: Qernel7 ok
