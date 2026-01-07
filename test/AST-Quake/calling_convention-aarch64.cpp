/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This test is only valid for aarch64.
// RUN: if [ `uname -m` = "aarch64" ] ; then \
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s ; fi

#include <cudaq.h>
#include <tuple>
#include <vector>

// Tests the host-side signatures of various spec supported kernel arguments and
// results. This file tests the x86_64 calling convention. Other architectures
// differ in their calling conventions.

//===----------------------------------------------------------------------===//
// test all the basic arithmetic types to deny any regressions.

struct T0 {
  void operator()() __qpu__ {}
};

struct T1 {
  void operator()(double arg) __qpu__ {}
};

struct T2 {
  void operator()(float arg) __qpu__ {}
};

struct T3 {
  void operator()(long long arg) __qpu__ {}
};

struct T4 {
  void operator()(long arg) __qpu__ {}
};

struct T5 {
  void operator()(int arg) __qpu__ {}
};

struct T6 {
  void operator()(short arg) __qpu__ {}
};

struct T7 {
  void operator()(char arg) __qpu__ {}
};

struct T8 {
  void operator()(bool arg) __qpu__ {}
};

// CHECK-LABEL:  func.func @_ZN2T0clEv(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>) {
// CHECK-LABEL:  func.func @_ZN2T1clEd(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: f64) {
// CHECK-LABEL:  func.func @_ZN2T2clEf(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: f32) {
// CHECK-LABEL:  func.func @_ZN2T3clEx(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i64) {
// CHECK-LABEL:  func.func @_ZN2T4clEl(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i64) {
// CHECK-LABEL:  func.func @_ZN2T5clEi(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i32) {
// CHECK-LABEL:  func.func @_ZN2T6clEs(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i16) {
// CHECK-LABEL:  func.func @_ZN2T7clEc(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i8) {
// CHECK-LABEL:  func.func @_ZN2T8clEb(
// CHECK-SAME:    %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i1) {

struct R0 {
  void operator()() __qpu__ {}
};

struct R1 {
  double operator()() __qpu__ { return {}; }
};

struct R2 {
  float operator()() __qpu__ { return {}; }
};

struct R3 {
  long long operator()() __qpu__ { return {}; }
};

struct R4 {
  long operator()() __qpu__ { return {}; }
};

struct R5 {
  int operator()() __qpu__ { return {}; }
};

struct R6 {
  short operator()() __qpu__ { return {}; }
};

struct R7 {
  char operator()() __qpu__ { return {}; }
};

struct R8 {
  bool operator()() __qpu__ { return {}; }
};

// CHECK-LABEL:  func.func @_ZN2R0clEv(%arg0: !cc.ptr<i8>) {
// CHECK-LABEL:  func.func @_ZN2R1clEv(%arg0: !cc.ptr<i8>) -> f64 {
// CHECK-LABEL:  func.func @_ZN2R2clEv(%arg0: !cc.ptr<i8>) -> f32 {
// CHECK-LABEL:  func.func @_ZN2R3clEv(%arg0: !cc.ptr<i8>) -> i64 {
// CHECK-LABEL:  func.func @_ZN2R4clEv(%arg0: !cc.ptr<i8>) -> i64 {
// CHECK-LABEL:  func.func @_ZN2R5clEv(%arg0: !cc.ptr<i8>) -> i32 {
// CHECK-LABEL:  func.func @_ZN2R6clEv(%arg0: !cc.ptr<i8>) -> i16 {
// CHECK-LABEL:  func.func @_ZN2R7clEv(%arg0: !cc.ptr<i8>) -> i8 {
// CHECK-LABEL:  func.func @_ZN2R8clEv(%arg0: !cc.ptr<i8>) -> i1 {

//===----------------------------------------------------------------------===//
// structs that are less than 128 bits.
// arguments may be merged into 1 register or passed in pair of registers.
// results are returned in registers.

struct G0 {
  std::pair<bool, bool> operator()(std::pair<double, double>) __qpu__ {
    return {};
  }
};

struct G1 {
  std::pair<bool, char> operator()(std::pair<float, float>) __qpu__ {
    return {};
  }
};

struct G2 {
  std::pair<char, short> operator()(std::pair<long, long>,
                                    std::pair<int, double>) __qpu__ {
    return {};
  }
};

struct G3 {
  std::pair<short, short> operator()(std::pair<double, bool>) __qpu__ {
    return {};
  }
};

struct BB {
  bool _1;
  bool _2;
  bool _3;
};

BB glue0();

struct G4 {
  std::pair<int, int> operator()(BB) __qpu__ { return {}; }
};

struct II {
  int _1;
  int _2;
  int _3;
};

II glue1();

struct G5 {
  std::pair<long, float> operator()(II) __qpu__ { return {}; }
};

struct CC {
  char _1;
  unsigned char _2;
  signed char _3;
};

CC glue2();

struct G6 {
  std::pair<long, long> operator()(CC) __qpu__ { return {}; }
};

struct G7 {
  BB operator()(BB, II, CC) __qpu__ { return glue0(); }
};

struct G8 {
  II operator()(II, CC, BB) __qpu__ { return glue1(); }
};

struct G9 {
  CC operator()(CC, BB, II) __qpu__ { return glue2(); }
};

// clang-format off
// CHECK-LABEL:  func.func @_ZN2G0clESt4pairIddE(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: !cc.array<f64 x 2>) -> i16
// CHECK-LABEL:  func.func @_ZN2G1clESt4pairIffE(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: !cc.array<f32 x 2>) -> i16
// CHECK-LABEL:  func.func @_ZN2G2clESt4pairIllES0_IidE(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: !cc.array<i64 x 2>, %[[VAL_2:.*]]: !cc.array<i64 x 2>) -> i32
// CHECK-LABEL:  func.func @_ZN2G3clESt4pairIdbE(
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>, %[[VAL_2:.*]]: !cc.array<i64 x 2>) -> i32
// CHECK-LABEL:  func.func @_ZN2G4clE2BB(
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>, %[[VAL_2:.*]]: i64) -> i64
// CHECK-LABEL:  func.func @_ZN2G5clE2II(
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>, %[[VAL_2:.*]]: !cc.array<i64 x 2>) -> !cc.array<i64 x 2>
// CHECK-LABEL:  func.func @_ZN2G6clE2CC(
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>, %[[VAL_2:.*]]: i64) -> !cc.array<i64 x 2>
// CHECK-LABEL:  func.func @_ZN2G7clE2BB2II2CC(
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>, %[[VAL_2:.*]]: i64,
// CHECK-SAME:     %[[VAL_3:.*]]: !cc.array<i64 x 2>, %[[VAL_4:.*]]: i64) -> i24
// CHECK-LABEL:  func.func @_ZN2G8clE2II2CC2BB(
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>, %[[VAL_2:.*]]: !cc.array<i64 x 2>,
// CHECK-SAME:     %[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64) -> !cc.array<i64 x 2>
// CHECK-LABEL:  func.func @_ZN2G9clE2CC2BB2II(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: !cc.array<i64 x 2>) -> i24
// clang-format on

//===----------------------------------------------------------------------===//
// std::vector - these get converted to sret and byval ptrs on host side.

std::vector<int> make_believe();

struct V0 {
  std::vector<int> operator()() __qpu__ { return make_believe(); }
};

std::vector<bool> make_coffee();

struct V1 {
  std::vector<bool> operator()(std::vector<double>) __qpu__ {
    return make_coffee();
  }
};

std::vector<std::pair<char, int>> make_crazy();

struct V2 {
  std::vector<std::pair<char, int>> operator()(std::vector<float>,
                                               std::vector<short>) __qpu__ {
    return make_crazy();
  }
};

struct V3 {
  void operator()(std::vector<long>, std::vector<bool>) __qpu__ {}
};

// clang-format off
// CHECK-LABEL:  func.func @_ZN2V0clEv(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<i32>, !cc.ptr<i32>, !cc.ptr<i32>}>> {llvm.sret = !cc.struct<{!cc.ptr<i32>, !cc.ptr<i32>, !cc.ptr<i32>}>},
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>)
// CHECK-LABEL:  func.func @_ZN2V1clESt6vectorIdSaIdEE(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<i1>, !cc.array<i8 x 32>}>> {llvm.sret = !cc.struct<{!cc.ptr<i1>, !cc.array<i8 x 32>}>},
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>,
// CHECK-SAME:     %[[VAL_2:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<f64>, !cc.ptr<f64>, !cc.ptr<f64>}>>)
// CHECK-LABEL:  func.func @_ZN2V2clESt6vectorIfSaIfEES0_IsSaIsEE(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<!cc.struct<{i8, i32} [64,4]>>, !cc.ptr<!cc.struct<{i8, i32} [64,4]>>, !cc.ptr<!cc.struct<{i8, i32} [64,4]>>}>> {llvm.sret = !cc.struct<{!cc.ptr<!cc.struct<{i8, i32} [64,4]>>, !cc.ptr<!cc.struct<{i8, i32} [64,4]>>, !cc.ptr<!cc.struct<{i8, i32} [64,4]>>}>},
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>,
// CHECK-SAME:     %[[VAL_2:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<f32>, !cc.ptr<f32>, !cc.ptr<f32>}>>,
// CHECK-SAME:     %[[VAL_3:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<i16>, !cc.ptr<i16>, !cc.ptr<i16>}>>)
// CHECK-LABEL:  func.func @_ZN2V3clESt6vectorIlSaIlEES0_IbSaIbEE(
// CHECK-SAME: %[[VAL_0:.*]]: !cc.ptr<i8>,
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<i64>, !cc.ptr<i64>, !cc.ptr<i64>}>>,
// CHECK-SAME:     %[[VAL_2:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<i1>, !cc.array<i8 x 32>}>>)
// clang-format on

//===----------------------------------------------------------------------===//
// structs that are more than 128 bits. These get converted to sret or byval
// ptrs on the host side.

struct B0 {
  void operator()(std::tuple<double, int, char, float, short>) __qpu__ {}
};

struct BG {
  float _1[4];
  int _2[5];
};

BG make_sausage();

struct B1 {
  BG operator()() __qpu__ { return make_sausage(); }
};

std::tuple<int, char, float, short, double, double> make_interesting();

struct B2 {
  std::tuple<int, char, float, short, double, double> operator()(BG) __qpu__ {
    return make_interesting();
  }
};

struct BA {
  bool _1[64];
};

struct B3 {
  BA operator()(BA arg) __qpu__ { return arg; }
};

// clang-format off
// CHECK-LABEL:  func.func @_ZN2B0clESt5tupleIJdicfsEE(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<i8>, %[[VAL_1:.*]]: !cc.ptr<!cc.struct<{i16, f32, i8, i32, f64}>>) {
// CHECK-LABEL:  func.func @_ZN2B1clEv(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<!cc.struct<"BG" {!cc.array<f32 x 4>, !cc.array<i32 x 5>} [288,4]>> {llvm.sret = !cc.struct<"BG" {!cc.array<f32 x 4>, !cc.array<i32 x 5>} [288,4]>},
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>)
// CHECK-LABEL:  func.func @_ZN2B2clE2BG(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<!cc.struct<{f64, f64, i16, f32, i8, i32}>> {llvm.sret = !cc.struct<{f64, f64, i16, f32, i8, i32}>},
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>,
// CHECK-SAME:     %[[VAL_2:.*]]: !cc.ptr<!cc.struct<"BG" {!cc.array<f32 x 4>, !cc.array<i32 x 5>} [288,4]>>)
// CHECK-LABEL:  func.func @_ZN2B3clE2BA(
// CHECK-SAME:     %[[VAL_0:.*]]: !cc.ptr<!cc.struct<"BA" {!cc.array<i1 x 64>} [512,1]>> {llvm.sret = !cc.struct<"BA" {!cc.array<i1 x 64>} [512,1]>},
// CHECK-SAME:     %[[VAL_1:.*]]: !cc.ptr<i8>,
// CHECK-SAME:     %[[VAL_2:.*]]: !cc.ptr<!cc.struct<"BA" {!cc.array<i1 x 64>} [512,1]>>)
// clang-format on
