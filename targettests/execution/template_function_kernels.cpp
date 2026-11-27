/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s

// Function templates marked `__qpu__`, specialized on types and on non-type
// arguments. Calling a specialization from the host must launch that
// specialization's kernel, and kernels can call one another.

#include <cstdio>
#include <cudaq.h>

template <typename T>
__qpu__ int byType(T t) {
  if constexpr (sizeof(T) > 2)
    return 10;
  else
    return 20;
}

template <bool B>
__qpu__ int byBool() {
  if constexpr (B)
    return 1;
  return 0;
}

template <int N>
__qpu__ int byInt() {
  return N;
}

template <typename T, int N>
__qpu__ int mixed(T t) {
  return N + 100;
}

template <typename... Ts>
__qpu__ int pack(Ts... ts) {
  return sizeof...(Ts);
}

// Fold expressions over the elements of a pack. Each element of the pack is a
// distinct parameter, and they must not be confused with one another.
template <typename... Ts>
__qpu__ int sum(Ts... ts) {
  return (ts + ...);
}
template <typename... Ts>
__qpu__ int sumInit(Ts... ts) {
  return (1 + ... + ts);
}
template <typename... Ts>
__qpu__ int leftSub(Ts... ts) {
  return (... - ts);
}
template <typename... Ts>
__qpu__ int rightSub(Ts... ts) {
  return (ts - ...);
}
template <typename... Ts>
__qpu__ bool all(Ts... ts) {
  return (... && ts);
}
template <typename... Ts>
__qpu__ int emptyFold(Ts... ts) {
  return (0 + ... + ts);
}

// Expansions of a pack as arguments and as an initializer list.
__qpu__ int three(int a, int b, int c) { return a * 100 + b * 10 + c; }
template <typename... Ts>
__qpu__ int forward(Ts... ts) {
  return three(ts...);
}
template <typename... Ts>
__qpu__ int doubled(Ts... ts) {
  return three((ts + ts)...);
}
template <typename... Ts>
__qpu__ int leading(Ts... ts) {
  return three(1, ts...);
}
template <typename... Ts>
__qpu__ int firstTwo(Ts... ts) {
  int a[] = {ts...};
  return a[0] + a[1];
}

// A fold over the comma operator, applied to qubits.
template <typename... Qs>
__qpu__ void flip(Qs &...qs) {
  (x(qs), ...);
}
__qpu__ void flipFirstAndLast() {
  cudaq::qubit a, b, c;
  flip(a, c);
  mz(a);
  mz(b);
  mz(c);
}

// A kernel that calls other specializations.
__qpu__ int nested() { return byInt<3>() + byInt<4>() + byBool<true>(); }

int main() {
  // The value each launch returns identifies the specialization that ran.
  printf("type int: %d\n", byType(1));
  printf("type char: %d\n", byType('c'));
  printf("bool true: %d\n", byBool<true>());
  printf("bool false: %d\n", byBool<false>());
  printf("int 3: %d\n", byInt<3>());
  printf("int 4: %d\n", byInt<4>());
  printf("mixed: %d\n", mixed<int, 5>(1));
  printf("pack: %d %d\n", pack(1, 'c'), pack(1.0, 2, 3u));
  printf("sum: %d\n", sum(1, 2, 3));
  printf("sumInit: %d\n", sumInit(4, 5));
  printf("leftSub: %d\n", leftSub(10, 1, 2));
  printf("rightSub: %d\n", rightSub(10, 1, 2));
  printf("all: %d %d\n", (int)all(true, true), (int)all(true, false));
  printf("emptyFold: %d\n", emptyFold());
  printf("forward: %d\n", forward(1, 2, 3));
  printf("doubled: %d\n", doubled(1, 2, 3));
  printf("leading: %d\n", leading(2, 3));
  printf("firstTwo: %d\n", firstTwo(1, 2, 3));
  for (auto &[bits, count] : cudaq::sample(flipFirstAndLast))
    printf("flip: %s\n", bits.c_str());
  printf("nested: %d\n", nested());
  return 0;
}

// CHECK: type int: 10
// CHECK: type char: 20
// CHECK: bool true: 1
// CHECK: bool false: 0
// CHECK: int 3: 3
// CHECK: int 4: 4
// CHECK: mixed: 105
// CHECK: pack: 2 3
// CHECK: sum: 6
// CHECK: sumInit: 10
// CHECK: leftSub: 7
// CHECK: rightSub: 11
// CHECK: all: 1 0
// CHECK: emptyFold: 0
// CHECK: forward: 123
// CHECK: doubled: 246
// CHECK: leading: 123
// CHECK: firstTwo: 3
// CHECK: flip: 101
// CHECK: nested: 8
