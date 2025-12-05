/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt -canonicalize -cse -lift-array-alloc -globalize-array-values -canonicalize | cudaq-translate | FileCheck %s
// clang-format on

#include <cudaq.h>

struct kernel {

  void operator()() __qpu__ {
    cudaq::qvector q(std::vector<cudaq::complex>({ M_SQRT1_2, M_SQRT1_2, 0., 0.}));
    [[maybe_unused]] auto result = mz(q);
  }
};

// clang-format off
// CHECK-LABEL: define void @__nvqpp__mlirgen__kernel() local_unnamed_addr {
// CHECK:         %[[VAL_0:.*]] = alloca [4 x { double, double }], align 8
// CHECK:         %[[VAL_1:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 0
// CHECK:         store double 0x3FE6A09E667F3BCD, double* %[[VAL_1]], align 8
// CHECK:         %[[VAL_2:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 1
// CHECK:         store double 0.000000e+00, double* %[[VAL_2]], align 8
// CHECK:         %[[VAL_3:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 0
// CHECK:         store double 0x3FE6A09E667F3BCD, double* %[[VAL_3]], align 8
// CHECK:         %[[VAL_4:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 1
// CHECK:         %[[VAL_5:.*]] = bitcast [4 x { double, double }]* %[[VAL_0]] to i8*
// CHECK:         %[[VAL_6:.*]] = bitcast double* %[[VAL_4]] to i8*
// CHECK:         call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(40) %[[VAL_6]], i8 0, i64 40, i1 false)
// CHECK:         %[[VAL_7:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_complex_f64(i8* nonnull %[[VAL_5]], i64 4)
// CHECK:         %[[VAL_8:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_7]])
// CHECK:         %[[VAL_9:.*]] = call %[[VAL_10:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_8]], i8** %[[VAL_7]])
// CHECK:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_7]])
// CHECK:         %[[VAL_11:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_10]]* %[[VAL_9]])
// CHECK:         %[[VAL_12:.*]] = icmp sgt i64 %[[VAL_11]], 0
// CHECK:         br i1 %[[VAL_12]], label %[[VAL_13:.*]], label %[[VAL_14:.*]]
// CHECK:       .{{.*}}:                                           ; preds = %[[VAL_15:.*]], %[[VAL_13]]
// CHECK:         %[[VAL_16:.*]] = phi i64 [ %[[VAL_17:.*]], %[[VAL_13]] ], [ 0, %[[VAL_15]] ]
// CHECK:         %[[VAL_18:.*]] = call %[[VAL_19:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_10]]* %[[VAL_9]], i64 %[[VAL_16]])
// CHECK:         %[[VAL_20:.*]] = load %[[VAL_19]]*, %[[VAL_19]]** %[[VAL_18]], align 8
// CHECK:         %[[VAL_21:.*]] = call %[[VAL_22:.*]]* @__quantum__qis__mz__to__register(%[[VAL_19]]* %[[VAL_20]], i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.726573756C7400, i64 0, i64 0))
// CHECK:         %[[VAL_17]] = add nuw nsw i64 %[[VAL_16]], 1
// CHECK:         %[[VAL_23:.*]] = icmp eq i64 %[[VAL_17]], %[[VAL_11]]
// CHECK:         br i1 %[[VAL_23]], label %[[VAL_14]], label %[[VAL_13]]
// CHECK:       .{{.*}}:                                      ; preds = %[[VAL_13]], %[[VAL_15]]
// CHECK:         call void @__quantum__rt__qubit_release_array(%[[VAL_10]]* %[[VAL_9]])
// CHECK:         ret void
// CHECK:       }
// clang-format on
