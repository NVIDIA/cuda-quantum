/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if [ command -v split-file ]; then \
// RUN: split-file %s %t && \
// RUN: nvq++ --enable-mlir -c %t/span_dumps.cpp -o %t/span_dumps.o && \
// RUN: nvq++ --enable-mlir -c %t/span_exercise.cpp -o %t/span_exercise.o && \
// RUN: nvq++ --enable-mlir %t/span_dumps.o %t/span_exercise.o -o %t/spanaroo.out && \
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

void dump_2d_int_vector(std::span<std::span<int>> x) {
  std::cout << "integer matrix: {\n";
  for (auto s : x) {
    std::cout << "    ";
    for (auto i : s)
      std::cout << i << "  ";
    std::cout << '\n';
  }
  std::cout << "}\n";
}

void dump_int_scalar(int x) { std::cout << "scalar integer: " << x << '\n'; }

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
void dump_int_scalar(int v);
void dump_bool_vector(const std::vector<bool> &pw);
void dump_double_vector(const std::vector<double> &pw);
void dump_2d_int_vector(const std::vector<std::vector<int>> &pw);
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

struct Interesting {
  std::vector<std::vector<std::vector<int>>> ragged3d;
  int flags;
  std::vector<double> angular;
};

__qpu__ void exciting(std::vector<Interesting> vi) {
  for (unsigned i = 0; i < vi.size(); ++i) {
    for (unsigned j = 0; j < vi[i].ragged3d.size(); ++j)
      dump_2d_int_vector(vi[i].ragged3d[j]);
    dump_int_scalar(vi[i].flags);
    dump_double_vector(vi[i].angular);
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
  kern4(vivp);

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
  qern4(vdvp);

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
  cern4(vbvp);

  std::vector<std::vector<int>> ix0 = {pw0, pw0};
  std::vector<std::vector<int>> ix1 = {pw1, pw0};
  std::vector<std::vector<int>> ix2 = {pw2, pw3, pw3};
  std::vector<std::vector<int>> ix3 = {{404}, {101, 202}};
  std::vector<std::vector<std::vector<int>>> i3d0 = {ix0, ix1};
  std::vector<std::vector<std::vector<int>>> i3d1 = {ix1};
  std::vector<std::vector<std::vector<int>>> i3d2 = {ix2, ix3};
  std::vector<std::vector<std::vector<int>>> i3d3 = {ix3};
  std::vector<std::vector<std::vector<int>>> i3d4 = {ix2, ix0, ix0};
  Interesting in0 = {i3d0, 66, {2.0, 4.0}};
  Interesting in1 = {i3d1, 123, {3.0, 6.0}};
  Interesting in2 = {i3d2, 561, {4.0, 8.0}};
  Interesting in3 = {i3d3, 72341, {5.0, 10.0}};
  Interesting in4 = {i3d4, -2348, {12.0, 5280.1}};
  std::vector<Interesting> ving = {in0, in1, in2, in3, in4};
  std::cout << "===\n";
  exciting(ving);

  return 0;
}

// CHECK: ---
// CHECK: integers: 345 1 2
// CHECK: ---
// CHECK: integers: 345 1 2
// CHECK-NEXT: integers: 92347 3 4
// CHECK-NEXT: integers: 2358 5 6
// CHECK-NEXT: integers: 45 7 18
// CHECK: ---
// CHECK: integers: 8 238 44
// CHECK-NEXT: integers: 0 -4 81 92745
// CHECK: ---
// CHECK: integers: 8 238 44
// CHECK-NEXT: integers: 0 -4 81 92745
// CHECK-NEXT: integers: 5 -87 43 1 76
// CHECK-NEXT: integers: 0 0 2 1
// CHECK-NEXT: integers: 1
// CHECK-NEXT: integers: -2 3
// CHECK-NEXT: integers: -4 -5 6
// CHECK-NEXT: integers: -7 -8 -9 88
// CHECK: ---
// CHECK: doubles: 3.45 1 2
// CHECK: ---
// CHECK: doubles: 3.45 1 2
// CHECK-NEXT: doubles: 92.347 2.3 4
// CHECK-NEXT: doubles: 235.8 5.5 6.4
// CHECK-NEXT: doubles: 4.5 77.7 18.2
// CHECK: ---
// CHECK: doubles: 8 2.38 4.4
// CHECK-NEXT: doubles: 0 -4.99 81.5 92.745
// CHECK: ---
// CHECK: doubles: 8 2.38 4.4
// CHECK-NEXT: doubles: 0 -4.99 81.5 92.745
// CHECK-NEXT: doubles: 5 -8.7 4.3 1 7.6
// CHECK-NEXT: doubles: 0 0 2 1
// CHECK-NEXT: doubles: 1
// CHECK-NEXT: doubles: -2 3
// CHECK-NEXT: doubles: -4 -5 6
// CHECK-NEXT: doubles: -7 -8 -9 0.88
// CHECK: ---
// CHECK: booleans: 1 0
// CHECK: ---
// CHECK: booleans: 1 0
// CHECK-NEXT: booleans: 0 0 0
// CHECK-NEXT: booleans: 0 1 0 1
// CHECK-NEXT: booleans: 0 0 1 0 1
// CHECK: ---
// CHECK: booleans: 0 0
// CHECK-NEXT: booleans: 0 1 1 0
// CHECK: ---
// CHECK: booleans: 0 0
// CHECK-NEXT: booleans: 0 1 1 0
// CHECK-NEXT: booleans: 0 1 1 0 1 0
// CHECK-NEXT: booleans: 0 1 1 0 0 0 1 0
// CHECK-NEXT: booleans: 0
// CHECK-NEXT: booleans: 1 1
// CHECK-NEXT: booleans: 1 0 0
// CHECK-NEXT: booleans: 0 1 0 1
// CHECK: ===
// CHECK: integer matrix: {
// CHECK-NEXT:     345  1  2
// CHECK-NEXT:     345  1  2
// CHECK-NEXT: }
// CHECK-NEXT: integer matrix: {
// CHECK-NEXT:     92347  3  4
// CHECK-NEXT:     345  1  2
// CHECK-NEXT: }
// CHECK-NEXT: scalar integer: 66
// CHECK-NEXT: doubles: 2 4
// CHECK-NEXT: integer matrix: {
// CHECK-NEXT:     92347  3  4
// CHECK-NEXT:     345  1  2
// CHECK-NEXT: }
// CHECK-NEXT: scalar integer: 123
// CHECK-NEXT: doubles: 3 6
// CHECK-NEXT: integer matrix: {
// CHECK-NEXT:     2358  5  6
// CHECK-NEXT:     45  7  18
// CHECK-NEXT:     45  7  18
// CHECK-NEXT: }
// CHECK-NEXT: integer matrix: {
// CHECK-NEXT:     404
// CHECK-NEXT:     101  202
// CHECK-NEXT: }
// CHECK-NEXT: scalar integer: 561
// CHECK-NEXT: doubles: 4 8
// CHECK-NEXT: integer matrix: {
// CHECK-NEXT:     404
// CHECK-NEXT:     101  202
// CHECK-NEXT: }
// CHECK-NEXT: scalar integer: 72341
// CHECK-NEXT: doubles: 5 10
// CHECK-NEXT: integer matrix: {
// CHECK-NEXT:     2358  5  6
// CHECK-NEXT:     45  7  18
// CHECK-NEXT:     45  7  18
// CHECK-NEXT: }
// CHECK-NEXT: integer matrix: {
// CHECK-NEXT:     345  1  2
// CHECK-NEXT:     345  1  2
// CHECK-NEXT: }
// CHECK-NEXT: integer matrix: {
// CHECK-NEXT:     345  1  2
// CHECK-NEXT:     345  1  2
// CHECK-NEXT: }
// CHECK-NEXT: scalar integer: -2348
// CHECK-NEXT: doubles: 12 5280.1
