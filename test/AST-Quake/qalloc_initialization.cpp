/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The next line is a workaround for bugs in the CI.
// REQUIRES: c++20

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt | cudaq-translate --convert-to=qir | FileCheck --check-prefix=QIR %s
// clang-format on

#include <cudaq.h>

struct Vanilla {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector v = {0., 1., 1., 0.};
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Vanilla() -> !cc.stdvec<i1>
// CHECK:           %[[VAL_4:.*]] = cc.address_of @__nvqpp__rodata_init_0 : !cc.ptr<!cc.array<f64 x 4>>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_4]] : (!quake.veq<4>, !cc.ptr<!cc.array<f64 x 4>>) -> !quake.veq<4>

// CHECK:         cc.global constant @__nvqpp__rodata_init_0 (dense<[0.0{{.*}}, 1.0{{.*}}, 1.0{{.*}}, 0.0{{.*}}]> : tensor<4xf64>) : !cc.array<f64 x 4>


// QIR-LABEL: @__nvqpp__rodata_init_0 = private constant [4 x double] [double 0.000000e+00, double 1.000000e+00, double 1.000000e+00, double 0.000000e+00]

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__Vanilla() local_unnamed_addr {
// QIR-NEXT:    %[[VAL_0:.*]] = tail call %Array* @__quantum__rt__qubit_allocate_array_with_state(i64 4, i8* nonnull bitcast ([4 x double]* @__nvqpp__rodata_init_0 to i8*))



