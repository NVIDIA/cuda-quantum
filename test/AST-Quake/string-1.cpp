/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s
// XFAIL: *

#include <cudaq.h>
#include <string>
#include <tuple>

void prepQubit(const std::string &basis, cudaq::qubit &q) __qpu__ {}

void RzArcTan2(bool input, std::string basis) __qpu__ {
  cudaq::qubit aux;
  cudaq::qubit resource;
  cudaq::qubit target;
  if (input) {
    x(target);
  }
  prepQubit(basis, target);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_prepQubit._Z9prepQubitRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERN5cudaq5quditILm2EEE(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.charspan, %[[VAL_1:.*]]: !quake.ref)
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_RzArcTan2._Z9RzArcTan2bNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
// CHECK-SAME:      %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: !cc.charspan) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_2:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           cc.if(%[[VAL_6]]) {
// CHECK:             quake.x %[[VAL_5]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           call @__nvqpp__mlirgen__function_prepQubit._Z9prepQubitRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERN5cudaq5quditILm2EEE(%[[VAL_1]], %[[VAL_5]]) : (!cc.charspan, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_Z9RzArcTan2bNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
// CHECK-SAME:      %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<i8>, i64, !cc.array<i8 x 16>}>>) attributes {no_this} {
// CHECK:           return
// CHECK:         }
// clang-format on
		
int main() {
  RzArcTan2(true, {});
  return 0;
}
