/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --opt-pass distributed-device-call %s -o %t && %t | FileCheck %s

// Regression test for the host calling convention of `device_call` callbacks,
// in particular the hidden parameters that the host ABI adds to a callback's
// signature: a hidden `sret` pointer for a result that is returned in memory,
// and `byval` for a struct that is passed in memory. Each callback prints what
// it received, so a misplaced or missing hidden parameter shows up as shifted
// or garbage values. The values are also checked after a round trip.
//
// The opposite direction, a kernel launched from the host, is covered by
// targettests/Kernel/signature-6.cpp.
//
// A callback result of type std::vector<std::vector<T>> is built here in the
// callback, i.e., in ordinary host code. Building one in the kernel is not
// supported by the frontend.

#include <cstdio>
#include <vector>

#include <cudaq.h>

// Too large to be passed or returned in registers. Passed `byval` on x86_64
// and returned via a hidden `sret` pointer.
struct Big {
  long a, b, c, d;
};

// Small enough to be passed in registers. On x86_64 it is passed as two
// separate integer arguments.
struct Two {
  long a, b;
};

// Dynamic, so it is never passed in registers. Passed by invisible reference
// and returned via a hidden `sret` pointer.
struct WithVec {
  int tag;
  std::vector<int> data;
};

void showBig(const char *who, Big b) {
  printf("%s: %ld %ld %ld %ld\n", who, b.a, b.b, b.c, b.d);
}

void showVec(const char *who, const std::vector<int> &v) {
  printf("%s:", who);
  for (auto x : v)
    printf(" %d", x);
  printf("\n");
}

// A `byval` struct as the only argument, as the first of two, and as the
// second of two.
void takeBig(Big b) { showBig("takeBig", b); }
void takeBigInt(Big b, int n) {
  showBig("takeBigInt", b);
  printf("takeBigInt n: %d\n", n);
}
void takeIntBig(int n, Big b) {
  printf("takeIntBig n: %d\n", n);
  showBig("takeIntBig", b);
}

// A `sret` result that is a static struct.
Big computeBig(Big b, int n) {
  showBig("computeBig", b);
  printf("computeBig n: %d\n", n);
  return {b.d + n, b.c + n, b.b + n, b.a + n};
}
void verifyBig(Big b) { showBig("verifyBig", b); }

// A `sret` result, with arguments of every ABI flavor around it.
std::vector<int> computeMixed(int a, std::vector<int> v, Big b, float f,
                              Big c) {
  printf("computeMixed a: %d f: %.1f\n", a, f);
  showVec("computeMixed v", v);
  showBig("computeMixed b", b);
  showBig("computeMixed c", c);
  std::vector<int> r = v;
  r.push_back(a);
  return r;
}
void verifyVec(std::vector<int> v) { showVec("verifyVec", v); }

// A `sret` result with structs that are passed in registers on either side of
// a struct that is passed in memory.
Big computeTwoBig(Two t, Big b, Two u) {
  printf("computeTwoBig t: %ld %ld u: %ld %ld\n", t.a, t.b, u.a, u.b);
  showBig("computeTwoBig b", b);
  return {t.a + u.a, t.b + u.b, b.a, b.d};
}

// A struct with a vector member, as both an argument and a `sret` result.
WithVec computeWithVec(WithVec w, double x) {
  printf("computeWithVec tag: %d x: %.1f\n", w.tag, x);
  showVec("computeWithVec data", w.data);
  WithVec r{w.tag + 1, {}};
  for (auto v : w.data)
    r.data.push_back(v * 10);
  return r;
}
void verifyWithVec(WithVec w) {
  printf("verifyWithVec tag: %d\n", w.tag);
  showVec("verifyWithVec data", w.data);
}

// A vector of vectors as a `sret` result.
std::vector<std::vector<int>> computeVecVec(int n) {
  printf("computeVecVec n: %d\n", n);
  std::vector<std::vector<int>> r;
  for (int i = 0; i < n; ++i)
    r.push_back(std::vector<int>(i + 1, i));
  return r;
}
void verifyVecVec(std::vector<std::vector<int>> v) {
  printf("verifyVecVec:");
  for (auto &inner : v) {
    printf(" [");
    for (auto x : inner)
      printf(" %d", x);
    printf(" ]");
  }
  printf("\n");
}

__qpu__ void test(WithVec w, Big big) {
  cudaq::qubit q;
  h(q);

  cudaq::device_call(takeBig, big);
  cudaq::device_call(takeBigInt, big, 5);
  cudaq::device_call(takeIntBig, 6, big);

  auto big2 = cudaq::device_call(computeBig, big, 100);
  cudaq::device_call(verifyBig, big2);

  std::vector<int> v{4, 5};
  auto vm = cudaq::device_call(computeMixed, 8, v, big, 1.5f, big2);
  cudaq::device_call(verifyVec, vm);

  Two t{10, 20};
  Two u{30, 40};
  auto big3 = cudaq::device_call(computeTwoBig, t, big, u);
  cudaq::device_call(verifyBig, big3);

  auto w2 = cudaq::device_call(computeWithVec, w, 2.5);
  cudaq::device_call(verifyWithVec, w2);

  auto vv = cudaq::device_call(computeVecVec, 3);
  cudaq::device_call(verifyVecVec, vv);
}

int main() {
  test(WithVec{3, {1, 2, 3}}, Big{1, 2, 3, 4});
  return 0;
}

// CHECK: takeBig: 1 2 3 4
// CHECK: takeBigInt: 1 2 3 4
// CHECK: takeBigInt n: 5
// CHECK: takeIntBig n: 6
// CHECK: takeIntBig: 1 2 3 4
// CHECK: computeBig: 1 2 3 4
// CHECK: computeBig n: 100
// CHECK: verifyBig: 104 103 102 101
// CHECK: computeMixed a: 8 f: 1.5
// CHECK: computeMixed v: 4 5
// CHECK: computeMixed b: 1 2 3 4
// CHECK: computeMixed c: 104 103 102 101
// CHECK: verifyVec: 4 5 8
// CHECK: computeTwoBig t: 10 20 u: 30 40
// CHECK: computeTwoBig b: 1 2 3 4
// CHECK: verifyBig: 40 60 1 4
// CHECK: computeWithVec tag: 3 x: 2.5
// CHECK: computeWithVec data: 1 2 3
// CHECK: verifyWithVec tag: 4
// CHECK: verifyWithVec data: 10 20 30
// CHECK: computeVecVec n: 3
// CHECK: verifyVecVec: [ 0 ] [ 1 1 ] [ 2 2 2 ]
