/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --opt-pass distributed-device-call %s -o %t && %t | FileCheck %s

// Functional-correctness regression test for the generalized,
// distributed-memory reference `device_call` lowering: verifies that
// arguments and results of a variety of types survive the round trip
// through the marshal/unmarshal communication buffer with the correct
// values, not just that the code compiles.
//
// Each type is exercised with a "compute" callback (whose argument is
// checked by printing it) followed by a "verify" callback that receives the
// compute callback's *return* value as its own argument. If either
// marshaling direction (argument or return) corrupted the value, this shows
// up in the printed output.
//
// std::vector<bool> is the interesting case here: on the host it is a
// bit-packed specialization, but on the device side it is represented as a
// plain span of bytes (like std::vector<char>). This confirms that a real
// std::vector<bool> comes back on the host side with correct values, not
// just that the buffer's bits happen to compile.
//
// std::vector<std::vector<T>> is supported by this lowering as both an
// argument and a (recursively, arbitrarily deep) return type, but is
// covered at the MLIR level in
// cudaq/test/Optimizer/distributed_device_call.qke rather than here, since
// building a std::vector<std::vector<T>> literal inside a __qpu__ kernel
// currently crashes cudaq-quake (a pre-existing, unrelated frontend
// limitation, not a marshaling issue). A struct containing a vector member
// does not hit that frontend limitation and is exercised below as a return
// type (computeStructVecRet); the argument direction is covered at the MLIR
// level (call_structvec_arg) for the same reason vector-of-vector is.

#include <cstdio>
#include <vector>

#include <cudaq.h>

bool computeBool(bool b) {
  printf("computeBool arg: %d\n", (int)b);
  return !b;
}
void verifyBool(bool b) { printf("verifyBool: %d\n", (int)b); }

char computeChar(char c) {
  printf("computeChar arg: %d\n", (int)c);
  return static_cast<char>(c + 1);
}
void verifyChar(char c) { printf("verifyChar: %d\n", (int)c); }

short computeShort(short s) {
  printf("computeShort arg: %d\n", (int)s);
  return static_cast<short>(s + 1);
}
void verifyShort(short s) { printf("verifyShort: %d\n", (int)s); }

int computeInt(int i) {
  printf("computeInt arg: %d\n", i);
  return i + 1;
}
void verifyInt(int i) { printf("verifyInt: %d\n", i); }

long computeLong(long l) {
  printf("computeLong arg: %ld\n", l);
  return l + 1;
}
void verifyLong(long l) { printf("verifyLong: %ld\n", l); }

long long computeLongLong(long long l) {
  printf("computeLongLong arg: %lld\n", l);
  return l + 1;
}
void verifyLongLong(long long l) { printf("verifyLongLong: %lld\n", l); }

float computeFloat(float f) {
  printf("computeFloat arg: %.1f\n", (double)f);
  return f + 1.0f;
}
void verifyFloat(float f) { printf("verifyFloat: %.1f\n", (double)f); }

double computeDouble(double d) {
  printf("computeDouble arg: %.1f\n", d);
  return d + 1.0;
}
void verifyDouble(double d) { printf("verifyDouble: %.1f\n", d); }

struct Pair {
  int a;
  double b;
};

Pair computeStruct(Pair p) {
  printf("computeStruct arg: %d %.1f\n", p.a, p.b);
  return Pair{p.a + 1, p.b + 1.0};
}
void verifyStruct(Pair p) { printf("verifyStruct: %d %.1f\n", p.a, p.b); }

std::vector<int> computeVecInt(std::vector<int> v) {
  printf("computeVecInt arg:");
  for (auto x : v)
    printf(" %d", x);
  printf("\n");
  std::vector<int> r;
  for (auto x : v)
    r.push_back(x * 2);
  return r;
}
void verifyVecInt(std::vector<int> v) {
  printf("verifyVecInt:");
  for (auto x : v)
    printf(" %d", x);
  printf("\n");
}

std::vector<double> computeVecDouble(std::vector<double> v) {
  printf("computeVecDouble arg:");
  for (auto x : v)
    printf(" %.1f", x);
  printf("\n");
  std::vector<double> r;
  for (auto x : v)
    r.push_back(x * 2.0);
  return r;
}
void verifyVecDouble(std::vector<double> v) {
  printf("verifyVecDouble:");
  for (auto x : v)
    printf(" %.1f", x);
  printf("\n");
}

std::vector<bool> computeVecBool(std::vector<bool> v) {
  printf("computeVecBool arg:");
  for (bool b : v)
    printf(" %d", (int)b);
  printf("\n");
  std::vector<bool> r;
  for (bool b : v)
    r.push_back(!b);
  return r;
}
void verifyVecBool(std::vector<bool> v) {
  printf("verifyVecBool:");
  for (bool b : v)
    printf(" %d", (int)b);
  printf("\n");
}

std::vector<Pair> computeVecStruct(std::vector<Pair> v) {
  printf("computeVecStruct arg:");
  for (auto &p : v)
    printf(" (%d,%.1f)", p.a, p.b);
  printf("\n");
  std::vector<Pair> r;
  for (auto &p : v)
    r.push_back(Pair{p.a + 1, p.b + 1.0});
  return r;
}
void verifyVecStruct(std::vector<Pair> v) {
  printf("verifyVecStruct:");
  for (auto &p : v)
    printf(" (%d,%.1f)", p.a, p.b);
  printf("\n");
}

struct WithVec {
  int tag;
  std::vector<int> data;
};

// A struct containing a vector member as a device_call *return* type. The
// real host return value (built entirely in ordinary host C++ here, not
// kernel code, so it does not hit the frontend limitation noted above) must
// be reduced into the buffer's dynamic-output-slot shape on the way out and
// reconstructed into a real device-side value on the way back in.
WithVec computeStructVecRet(int tag) {
  printf("computeStructVecRet arg: %d\n", tag);
  WithVec w;
  w.tag = tag;
  std::vector<int> d{1, 2, 3};
  w.data = d;
  return w;
}
void verifyStructVecRet(WithVec w) {
  printf("verifyStructVecRet: tag=%d data:", w.tag);
  for (auto x : w.data)
    printf(" %d", x);
  printf("\n");
}

__qpu__ void test() {
  cudaq::qubit q;
  h(q);

  auto b = cudaq::device_call(computeBool, true);
  cudaq::device_call(verifyBool, b);

  auto c = cudaq::device_call(computeChar, static_cast<char>(5));
  cudaq::device_call(verifyChar, c);

  auto s = cudaq::device_call(computeShort, static_cast<short>(10));
  cudaq::device_call(verifyShort, s);

  auto i = cudaq::device_call(computeInt, 42);
  cudaq::device_call(verifyInt, i);

  auto l = cudaq::device_call(computeLong, 100L);
  cudaq::device_call(verifyLong, l);

  auto ll = cudaq::device_call(computeLongLong, 1000LL);
  cudaq::device_call(verifyLongLong, ll);

  auto f = cudaq::device_call(computeFloat, 1.5f);
  cudaq::device_call(verifyFloat, f);

  auto d = cudaq::device_call(computeDouble, 2.5);
  cudaq::device_call(verifyDouble, d);

  Pair p;
  p.a = 7;
  p.b = 3.5;
  auto p2 = cudaq::device_call(computeStruct, p);
  cudaq::device_call(verifyStruct, p2);

  std::vector<int> vi{1, 2, 3};
  auto vi2 = cudaq::device_call(computeVecInt, vi);
  cudaq::device_call(verifyVecInt, vi2);

  std::vector<double> vd{1.5, 2.5};
  auto vd2 = cudaq::device_call(computeVecDouble, vd);
  cudaq::device_call(verifyVecDouble, vd2);

  std::vector<bool> vb{true, false, true};
  auto vb2 = cudaq::device_call(computeVecBool, vb);
  cudaq::device_call(verifyVecBool, vb2);

  Pair vp0;
  vp0.a = 1;
  vp0.b = 1.0;
  Pair vp1;
  vp1.a = 2;
  vp1.b = 2.0;
  std::vector<Pair> vp{vp0, vp1};
  auto vp2 = cudaq::device_call(computeVecStruct, vp);
  cudaq::device_call(verifyVecStruct, vp2);

  auto w = cudaq::device_call(computeStructVecRet, 7);
  cudaq::device_call(verifyStructVecRet, w);
}

int main() {
  test();
  return 0;
}

// CHECK: computeBool arg: 1
// CHECK: verifyBool: 0
// CHECK: computeChar arg: 5
// CHECK: verifyChar: 6
// CHECK: computeShort arg: 10
// CHECK: verifyShort: 11
// CHECK: computeInt arg: 42
// CHECK: verifyInt: 43
// CHECK: computeLong arg: 100
// CHECK: verifyLong: 101
// CHECK: computeLongLong arg: 1000
// CHECK: verifyLongLong: 1001
// CHECK: computeFloat arg: 1.5
// CHECK: verifyFloat: 2.5
// CHECK: computeDouble arg: 2.5
// CHECK: verifyDouble: 3.5
// CHECK: computeStruct arg: 7 3.5
// CHECK: verifyStruct: 8 4.5
// CHECK: computeVecInt arg: 1 2 3
// CHECK: verifyVecInt: 2 4 6
// CHECK: computeVecDouble arg: 1.5 2.5
// CHECK: verifyVecDouble: 3.0 5.0
// CHECK: computeVecBool arg: 1 0 1
// CHECK: verifyVecBool: 0 1 0
// CHECK: computeVecStruct arg: (1,1.0) (2,2.0)
// CHECK: verifyVecStruct: (2,2.0) (3,3.0)
// CHECK: computeStructVecRet arg: 7
// CHECK: verifyStructVecRet: tag=7 data: 1 2 3
