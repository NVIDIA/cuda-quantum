/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --lower-to-cfg | cudaq-translate --convert-to=qir | FileCheck %s
// clang-format on

#include <cudaq.h>

struct kernel {
  void operator()() __qpu__ {
    cudaq::qarray<3> q;
    h(q[1]);
    x<cudaq::ctrl>(q[1], q[2]);

    x<cudaq::ctrl>(q[0], q[1]);
    h(q[0]);

    auto b0 = mz(q[0]);
    auto b1 = mz(q[1]);

    if (b1)
      x(q[2]);
    if (b0)
      z(q[2]);
  }
};

// clang-format off
// CHECK-LABEL: define void @__nvqpp__mlirgen__kernel()
// CHECK:         %[[VAL_0:.*]] = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 3)
// CHECK:         %[[VAL_2:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 1)
// CHECK:         %[[VAL_4:.*]] = load %Qubit*, %Qubit** %[[VAL_2]], align 8
// CHECK:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_4]])
// CHECK:         %[[VAL_5:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 2)
// CHECK:         %[[VAL_6:.*]] = load %Qubit*, %Qubit** %[[VAL_5]], align 8
// CHECK:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_4]], %Qubit* %[[VAL_6]])
// CHECK:         %[[VAL_7:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 0)
// CHECK:         %[[VAL_8:.*]] = load %Qubit*, %Qubit** %[[VAL_7]], align 8
// CHECK:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_8]], %Qubit* %[[VAL_4]])
// CHECK:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_8]])
// CHECK:         %[[VAL_9:.*]] = tail call %Result* @__quantum__qis__mz__to__register(%Qubit* %[[VAL_8]], i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.623000, i64 0, i64 0))
// CHECK:         %[[VAL_11:.*]] = bitcast %Result* %[[VAL_9]] to i1*
// CHECK:         %[[VAL_12:.*]] = load i1, i1* %[[VAL_11]], align 1
// CHECK:         %[[VAL_13:.*]] = tail call %Result* @__quantum__qis__mz__to__register(%Qubit* %[[VAL_4]], i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.623100, i64 0, i64 0))
// CHECK:         %[[VAL_14:.*]] = bitcast %Result* %[[VAL_13]] to i1*
// CHECK:         %[[VAL_15:.*]] = load i1, i1* %[[VAL_14]], align 1
// CHECK:         br i1 %[[VAL_15]], label %[[VAL_16:.*]], label %[[VAL_17:.*]]
// CHECK:       14:                                               ; preds = %[[VAL_18:.*]]
// CHECK:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_6]])
// CHECK:         br label %[[VAL_17]]
// CHECK:       15:                                               ; preds = %[[VAL_16]], %[[VAL_18]]
// CHECK:         br i1 %[[VAL_12]], label %[[VAL_19:.*]], label %[[VAL_20:.*]]
// CHECK:       16:                                               ; preds = %[[VAL_17]]
// CHECK:         tail call void @__quantum__qis__z(%Qubit* %[[VAL_6]])
// CHECK:         br label %[[VAL_20]]
// CHECK:       17:                                               ; preds = %[[VAL_19]], %[[VAL_17]]
// CHECK:         tail call void @__quantum__rt__qubit_release_array(%Array* %[[VAL_0]])
// CHECK:         ret void
// CHECK:       }

