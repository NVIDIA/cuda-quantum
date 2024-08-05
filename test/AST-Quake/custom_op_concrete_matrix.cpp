/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt -const-prop-complex -lift-array-value -get-concrete-matrix | FileCheck %s

#include <cudaq.h>

CUDAQ_REGISTER_OPERATION(custom_h, 1, 0,
                         {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2})

CUDAQ_REGISTER_OPERATION(custom_cnot, 2, 0,
                         {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0})


__qpu__ void kernel_1() {
  cudaq::qubit q, r;
  custom_h(q);
  custom_cnot(q, r);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_1._Z8kernel_1v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_h_generator_1._Z20custom_h_generator_{{.*}}vectorId{{.*}}.rodata_{{[0-9]+}} %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_cnot_generator_2._Z23custom_cnot_generator_{{.*}}vectorId{{.*}}.rodata_{{[0-9]+}} %[[VAL_0]], %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK:         cc.global constant @__nvqpp__mlirgen__function_custom_h_generator_1._Z20custom_h_generator_{{.*}}vectorId{{.*}}.rodata_{{[0-9]+}} (dense<[(0.70710678118654757,0.000000e+00), (0.70710678118654757,0.000000e+00), (0.70710678118654757,0.000000e+00), (-0.70710678118654757,0.000000e+00)]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
// CHECK:         cc.global constant @__nvqpp__mlirgen__function_custom_cnot_generator_2._Z23custom_cnot_generator_{{.*}}vectorId{{.*}}.rodata_{{[0-9]+}} (dense<[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<16xcomplex<f64>>) : !cc.array<complex<f64> x 16>
