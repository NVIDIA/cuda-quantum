/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if [ command -v split-file ]; then \
// RUN: split-file %s %t && \
// RUN: nvq++ %cpp_std --enable-mlir -c %t/span_dumps.cpp -o %t/span_dumps.o && \
// RUN: nvq++ %cpp_std --enable-mlir -c %t/span_exercise.cpp -o %t/span_exercise.o && \
// RUN: nvq++ %cpp_std --enable-mlir %t/span_dumps.o %t/span_exercise.o -o %t/spanaroo.out && \
// RUN: %t/spanaroo.out | FileCheck %s ; else \
// RUN: echo "skipping" ; fi
// clang-format on

//--- span_dumps.cpp

#include <iostream>
#include <span>
#include <string>

extern "C" {
void dump_bool_vector(std::span<bool> x) {
  std::cout << "booleans: ";
  for (auto i : x)
    std::cout << i << ' ';
  std::cout << '\n';
}

void dump_int_vector(std::span<int> x) {
  std::cout << "integers: ";
  for (auto i : x)
    std::cout << i << ' ';
  std::cout << '\n';
}

void dump_double_vector(std::span<double> x) {
  std::cout << "doubles: ";
  for (auto d : x)
    std::cout << d << ' ';
  std::cout << '\n';
}
}

//--- span_exercise.cpp

#include <cudaq.h>
#include <iostream>

// Fake host C++ signature that matches.
extern "C" {
void dump_int_vector(const std::vector<int> &pw);
void dump_bool_vector(const std::vector<bool> &pw);
void dump_double_vector(const std::vector<double> &pw);
}

__qpu__ void kern1(std::vector<int> arg) { dump_int_vector(arg); }

__qpu__ void kern2(std::vector<std::vector<int>> arg) {
  for (unsigned i = 0; i < arg.size(); ++i)
    dump_int_vector(arg[i]);
}

struct IntVectorPair {
  std::vector<int> _0;
  std::vector<int> _1;
};

__qpu__ void kern3(IntVectorPair ivp) {
  dump_int_vector(ivp._0);
  dump_int_vector(ivp._1);
}

__qpu__ void kern4(std::vector<IntVectorPair> vivp) {
  for (unsigned i = 0; i < vivp.size(); ++i) {
    dump_int_vector(vivp[i]._0);
    dump_int_vector(vivp[i]._1);
  }
}

__qpu__ void qern1(std::vector<double> arg) { dump_double_vector(arg); }

__qpu__ void qern2(std::vector<std::vector<double>> arg) {
  for (unsigned i = 0; i < arg.size(); ++i)
    dump_double_vector(arg[i]);
}

struct DoubleVectorPair {
  std::vector<double> _0;
  std::vector<double> _1;
};

__qpu__ void qern3(DoubleVectorPair ivp) {
  dump_double_vector(ivp._0);
  dump_double_vector(ivp._1);
}

__qpu__ void qern4(std::vector<DoubleVectorPair> vivp) {
  for (unsigned i = 0; i < vivp.size(); ++i) {
    dump_double_vector(vivp[i]._0);
    dump_double_vector(vivp[i]._1);
  }
}

__qpu__ void cern1(std::vector<bool> arg) { dump_bool_vector(arg); }

__qpu__ void cern2(std::vector<std::vector<bool>> arg) {
  for (unsigned i = 0; i < arg.size(); ++i)
    dump_bool_vector(arg[i]);
}

struct BoolVectorPair {
  std::vector<bool> _0;
  std::vector<bool> _1;
};

__qpu__ void cern3(BoolVectorPair ivp) {
  dump_bool_vector(ivp._0);
  dump_bool_vector(ivp._1);
}

__qpu__ void cern4(std::vector<BoolVectorPair> vivp) {
  for (unsigned i = 0; i < vivp.size(); ++i) {
    dump_bool_vector(vivp[i]._0);
    dump_bool_vector(vivp[i]._1);
  }
}

int main() {
  std::vector<int> pw0 = {345, 1, 2};
  std::cout << "---\n";
  kern1(pw0);
  std::vector<int> pw1 = {92347, 3, 4};
  std::vector<int> pw2 = {2358, 5, 6};
  std::vector<int> pw3 = {45, 7, 18};
  std::vector<std::vector<int>> vpw{pw0, pw1, pw2, pw3};
  std::cout << "---\n";
  kern2(vpw);

  IntVectorPair ivp = {{8, 238, 44}, {0, -4, 81, 92745}};
  std::cout << "---\n";
  kern3(ivp);

  IntVectorPair ivp2 = {{5, -87, 43, 1, 76}, {0, 0, 2, 1}};
  IntVectorPair ivp3 = {{1}, {-2, 3}};
  IntVectorPair ivp4 = {{-4, -5, 6}, {-7, -8, -9, 88}};
  std::vector<IntVectorPair> vivp = {ivp, ivp2, ivp3, ivp4};
  std::cout << "---\n";
  // kern4(vivp);

  std::vector<double> dpw0 = {3.45, 1., 2.};
  std::cout << "---\n";
  qern1(dpw0);
  std::vector<double> dpw1 = {92.347, 2.3, 4.};
  std::vector<double> dpw2 = {235.8, 5.5, 6.4};
  std::vector<double> dpw3 = {4.5, 77.7, 18.2};
  std::vector<std::vector<double>> vdpw{dpw0, dpw1, dpw2, dpw3};
  std::cout << "---\n";
  qern2(vdpw);

  DoubleVectorPair dvp = {{8., 2.38, 4.4}, {0., -4.99, 81.5, 92.745}};
  std::cout << "---\n";
  qern3(dvp);

  DoubleVectorPair dvp2 = {{5., -8.7, 4.3, 1., 7.6}, {0., 0., 2., 1.}};
  DoubleVectorPair dvp3 = {{1.}, {-2., 3.}};
  DoubleVectorPair dvp4 = {{-4., -5., 6.}, {-7., -8., -9., .88}};
  std::vector<DoubleVectorPair> vdvp = {dvp, dvp2, dvp3, dvp4};
  std::cout << "---\n";
  // qern4(vdvp);

  std::vector<bool> bpw0 = {true, false};
  std::cout << "---\n";
  cern1(bpw0);
  std::vector<bool> bpw1 = {false, false, false};
  std::vector<bool> bpw2 = {false, true, false, true};
  std::vector<bool> bpw3 = {false, false, true, false, true};
  std::vector<std::vector<bool>> vbpw{bpw0, bpw1, bpw2, bpw3};
  std::cout << "---\n";
  cern2(vbpw);

  BoolVectorPair bvp = {{false, false}, {false, true, true, false}};
  std::cout << "---\n";
  cern3(bvp);

  BoolVectorPair bvp2 = {{false, true, true, false, true, false},
                         {false, true, true, false, false, false, true, false}};
  BoolVectorPair bvp3 = {{false}, {true, true}};
  BoolVectorPair bvp4 = {{true, false, false}, {false, true, false, true}};
  std::vector<BoolVectorPair> vbvp = {bvp, bvp2, bvp3, bvp4};
  std::cout << "---\n";
  // cern4(vbvp);

  return 0;
}

// CHECK: ---
// CHECK: integers: 345 1 2
// CHECK: ---
// CHECK: integers: 345 1 2
// CHECK: integers: 92347 3 4
// CHECK: integers: 2358 5 6
// CHECK: integers: 45 7 18
// CHECK: ---
// CHECK: integers: 8 238 44
// CHECK: integers: 0 -4 81 92745
// CHECK: ---
// CHECK: doubles: 3.45 1 2
// CHECK: ---
// CHECK: doubles: 3.45 1 2
// CHECK: doubles: 92.347 2.3 4
// CHECK: doubles: 235.8 5.5 6.4
// CHECK: doubles: 4.5 77.7 18.2
// CHECK: ---
// CHECK: doubles: 8 2.38 4.4
// CHECK: doubles: 0 -4.99 81.5 92.745
// CHECK: ---
// CHECK: booleans: 1 0
// CHECK: ---
// CHECK: booleans: 1 0
// CHECK: booleans: 0 0 0
// CHECK: booleans: 0 1 0 1
// CHECK: booleans: 0 0 1 0 1
// CHECK: ---
// CHECK: booleans: 0 0
// CHECK: booleans: 0 1 1 0
