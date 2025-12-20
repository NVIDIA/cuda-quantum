/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iomanip>
#include <iostream>

// Tests that we can take a small struct, a struct with a vector member, a
// vector of small structs, and a large struct as an argument and return the
// same.

#define NYI /*__qpu__*/

void ok() { std::cout << "ok\n"; }
void fail() { std::cout << "fail\n"; }

struct S1 {
  int _1;
  double _2;
};

class QernelS1a {
public:
  void operator()(S1 s) __qpu__ {
    if (s._1 == 4 && s._2 == 8.2)
      ok();
    else
      fail();
  }
};

struct QernelS1 {
  S1 operator()(S1 s) NYI {
    if (s._1 == 4 && s._2 == 8.2)
      ok();
    else
      fail();
    return {++s._1, 0.0};
  }
};

struct S2 {
  int _1;
  std::vector<float> _2;
  double _3;
};

struct QernelS2a {
  void operator()(S2 s) __qpu__ {
    if (s._1 == 6 && s._2.size() == 2 && s._2[0] == 0.10f && s._2[1] == 0.93f &&
        s._3 == 16.4)
      ok();
    else
      fail();
  }
};

struct QernelS2 {
  // kernel result type not supported (bridge)
  S2 operator()(S2 s) NYI {
    s._1++;
    s._2[0] = 0.0;
    s._3 = -s._3;
    return s;
  }
};

class QernelS3a {
public:
  void operator()(std::vector<S1> s) __qpu__ {
    if (s[0]._1 == 4 && s[0]._2 == 8.2)
      ok();
    else
      fail();
  }
};

struct QernelS3 {
  std::vector<S1> operator()(std::vector<S1> s) __qpu__ {
    s[0]._1++;
    s[0]._2 = 0.0;
    return s;
  }
};

std::vector<S1> mock_ctor(const std::vector<S1> &v) { return v; }

struct QernelS4 {
  std::vector<S1> operator()(std::vector<S1> s) NYI {
    s[0]._1++;
    s[0]._2 = 0.0;
    return mock_ctor(s);
  }
};

struct S5 {
  int _1;
  double _2;
  char _3;
  double _4;
  float _5;
};

class QernelS5a {
public:
  void operator()(S5 s) __qpu__ {
    if (s._1 == 999 && s._2 == 17.76 && s._3 == 65 && s._4 == 18.12 &&
        s._5 == 19.19f)
      ok();
    else
      fail();
  }
};

class QernelS5 {
public:
  S5 operator()(S5 s) __qpu__ { return {s._1, s._4, 81, -s._2, 0.0f}; }
};

int main() {
  S1 s1 = {4, 8.2};
  std::cout << "QernelS1a ";
  QernelS1a{}(s1);
  std::cout << "QernelS1 ";
  auto updated_s1 = QernelS1{}(s1);
  if (updated_s1._1 == 5 && updated_s1._2 == 0.0)
    ok();
  else
    fail();

  std::vector<float> v = {0.10f, 0.93f};
  S2 s2 = {6, v, 16.4};
  std::cout << "QernelS2a ";
  QernelS2a{}(s2);
  std::cout << "QernelS2 ";
  auto updated_s2 = QernelS2{}(s2);
  if (updated_s2._1 == 7 && updated_s2._2[0] == 0.0f &&
      updated_s2._2[1] == 0.93f && updated_s2._3 == -s2._3)
    ok();
  else
    fail();

  std::vector<S1> vs1 = {s1, updated_s1};
  std::cout << "QernelS3a ";
  QernelS3a{}(vs1);
  std::cout << "QernelS3 ";
  auto updated_vs1 = QernelS3{}(vs1);
  if (updated_vs1[0]._1 == 5 && updated_vs1[0]._2 == 0.0 &&
      updated_vs1[1]._1 == updated_s1._1 && updated_vs1[1]._2 == updated_s1._2)
    ok();
  else
    fail();

  s1._1 = 88;
  std::vector<S1> vs2 = {s1, updated_s1};
  std::cout << "QernelS4 ";
  auto updated_vs2 = QernelS4{}(vs2);
  if (updated_vs2[0]._1 == 89 && updated_vs2[0]._2 == 0.0 &&
      updated_vs2[1]._1 == updated_s1._1 && updated_vs2[1]._2 == updated_s1._2)
    ok();
  else
    fail();

  S5 s5 = {999, 17.76, 'A', 18.12, 19.19f};
  std::cout << "QernelS5a ";
  QernelS5a{}(s5);
  std::cout << "QernelS5 ";
  auto updated_s5 = QernelS5{}(s5);
  if (updated_s5._1 == s5._1 && updated_s5._2 == s5._4 &&
      updated_s5._3 == 'Q' && updated_s5._4 == -s5._2 && updated_s5._5 == 0.0f)
    ok();
  else
    fail();

  return 0;
}

// CHECK-LABEL: QernelS1a ok
// CHECK-NEXT: QernelS1 ok
// CHECK-NEXT: ok
// CHECK-NEXT: QernelS2a ok
// CHECK-NEXT: QernelS2 ok
// CHECK-NEXT: QernelS3a ok
// CHECK-NEXT: QernelS3 ok
// CHECK-NEXT: QernelS4 ok
// CHECK-NEXT: QernelS5a ok
// CHECK-NEXT: QernelS5 ok
