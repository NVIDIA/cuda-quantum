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

// Test kernels can take arguments of tuple or pair as well as return values of
// same.

// FIXME: tuple and pair are not handled.
#define NYI /*__qpu__*/

void ok() { std::cout << "ok\n"; }
void fail() { std::cout << "fail\n"; }

using S1 = std::tuple<int, int, int>;

struct QernelS1a {
  void operator()(S1 s) NYI {
    if (std::get<0>(s) == 1 && std::get<1>(s) == 2 && std::get<2>(s) == 4)
      ok();
    else
      fail();
  }
};

struct QernelS1 {
  S1 operator()(S1 s) NYI {
    return {std::get<2>(s), std::get<1>(s), std::get<0>(s)};
  }
};

using S2 = std::tuple<double, float, std::vector<int>>;

struct QernelS2a {
  void operator()(S2 s) NYI {
    if (std::get<0>(s) == 8.16 && std::get<1>(s) == 32.64f &&
        std::get<2>(s).size() == 2)
      ok();
    else
      fail();
  }
};

struct QernelS2 {
  S2 operator()(S2 s) NYI {
    auto &v = std::get<2>(s);
    if (v[0] == 128 && v[1] == 256)
      ok();
    else
      fail();
    std::vector<int> k = {512, 1024};
    return {std::get<0>(s), std::get<1>(s), k};
  }
};

using S3 = std::pair<char, short>;

struct QernelS3a {
  void operator()(S3 s) NYI {
    if (s.first == 'c' && s.second == 12)
      ok();
    else
      fail();
  }
};

struct QernelS3 {
  S3 operator()(S3 s) NYI { return {s.first + 1, s.second - 1}; }
};

int main() {
  S1 s1{1, 2, 4};
  std::cout << "QernelS1a ";
  QernelS1a{}(s1);
  std::cout << "QernelS1 ";
  auto updated_s1 = QernelS1{}(s1);
  if (std::get<0>(updated_s1) == 4 && std::get<1>(updated_s1) == 2 &&
      std::get<2>(updated_s1) == 1)
    ok();
  else
    fail();

  std::vector<int> v = {128, 256};
  S2 s2 = {8.16, 32.64f, v};
  std::cout << "QernelS2a ";
  QernelS2a{}(s2);
  std::cout << "QernelS2 ";
  auto updated_s2 = QernelS2{}(s2);
  auto &r2 = std::get<2>(updated_s2);
  if (r2[0] == 512 && r2[1] == 1024)
    ok();
  else
    fail();

  S3 s3{'c', 12};
  std::cout << "QernelS3a ";
  QernelS3a{}(s3);
  std::cout << "QernelS3 ";
  auto updated_vs1 = QernelS3{}(s3);
  if (updated_vs1.first == 'd' && updated_vs1.second == 11)
    ok();
  else
    fail();

  return 0;
}

// clang-format off
// CHECK-LABEL: QernelS1a ok
// CHECK-NEXT: QernelS1 ok
// CHECK-NEXT: QernelS2a ok
// CHECK-NEXT: QernelS2 ok
// CHECK-NEXT: ok
// CHECK-NEXT: QernelS3a ok
// CHECK-NEXT: QernelS3 ok
