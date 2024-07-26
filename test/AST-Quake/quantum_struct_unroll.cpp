/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt --unrolling-pipeline | FileCheck %s

#include "cudaq.h"

struct test {
  int i;
  double d;
  cudaq::qview<> q;
};

__qpu__ void entry() {
  cudaq::qvector q(4);
  test tt{4, 2.2, q};
  h(tt.q);
}

int main() { cudaq::sample(entry); }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_entry._Z5entryv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][2] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_0]][3] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.h %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }