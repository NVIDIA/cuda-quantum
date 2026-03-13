/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
// CHECK:         %[[VAL_0:.*]] = tail call %[[VAL_1:.*]]* @__quantum__rt__qubit_allocate_array(i64 3)
// CHECK:         %[[VAL_2:.*]] = tail call %[[VAL_3:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_1]]* %[[VAL_0]], i64 1)
// CHECK:         %[[VAL_4:.*]] = load %[[VAL_3]]*, %[[VAL_3]]** %[[VAL_2]], align 8
// CHECK:         tail call void @__quantum__qis__h(%[[VAL_3]]* %[[VAL_4]])
// CHECK:         %[[VAL_5:.*]] = tail call %[[VAL_3]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_1]]* %[[VAL_0]], i64 2)
// CHECK:         %[[VAL_6:.*]] = load %[[VAL_3]]*, %[[VAL_3]]** %[[VAL_5]], align 8
// CHECK:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%[[VAL_1]]*, %[[VAL_3]]*)* @__quantum__qis__x__ctl to i8*), %[[VAL_3]]* %[[VAL_4]], %[[VAL_3]]* %[[VAL_6]])
// CHECK:         %[[VAL_7:.*]] = tail call %[[VAL_3]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_1]]* %[[VAL_0]], i64 0)
// CHECK:         %[[VAL_8:.*]] = load %[[VAL_3]]*, %[[VAL_3]]** %[[VAL_7]], align 8
// CHECK:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%[[VAL_1]]*, %[[VAL_3]]*)* @__quantum__qis__x__ctl to i8*), %[[VAL_3]]* %[[VAL_8]], %[[VAL_3]]* %[[VAL_4]])
// CHECK:         tail call void @__quantum__qis__h(%[[VAL_3]]* %[[VAL_8]])
// CHECK:         %[[VAL_9:.*]] = tail call %[[VAL_10:.*]]* @__quantum__qis__mz__to__register(%[[VAL_3]]* %[[VAL_8]], i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.623000, i64 0, i64 0))
// CHECK:         %[[VAL_11:.*]] = tail call %[[VAL_10]]* @__quantum__qis__mz__to__register(%[[VAL_3]]* %[[VAL_4]], i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.623100, i64 0, i64 0))
// CHECK:         %[[VAL_12:.*]] = bitcast %[[VAL_10]]* %[[VAL_11]] to i1*
// CHECK:         %[[VAL_13:.*]] = load i1, i1* %[[VAL_12]], align 1
// CHECK:         br i1 %[[VAL_13]], label %[[VAL_14:.*]], label %[[VAL_15:.*]]
// CHECK:       12:                                               ; preds = %[[VAL_16:.*]]
// CHECK:         tail call void @__quantum__qis__x(%[[VAL_3]]* %[[VAL_6]])
// CHECK:         br label %[[VAL_15]]
// CHECK:       13:                                               ; preds = %[[VAL_14]], %[[VAL_16]]
// CHECK:         %[[VAL_17:.*]] = bitcast %[[VAL_10]]* %[[VAL_9]] to i1*
// CHECK:         %[[VAL_18:.*]] = load i1, i1* %[[VAL_17]], align 1
// CHECK:         br i1 %[[VAL_18]], label %[[VAL_19:.*]], label %[[VAL_20:.*]]
// CHECK:       16:                                               ; preds = %[[VAL_15]]
// CHECK:         tail call void @__quantum__qis__z(%[[VAL_3]]* %[[VAL_6]])
// CHECK:         br label %[[VAL_20]]
// CHECK:       17:                                               ; preds = %[[VAL_19]], %[[VAL_15]]
// CHECK:         tail call void @__quantum__rt__qubit_release_array(%[[VAL_1]]* %[[VAL_0]])
// CHECK:         ret void
// CHECK:       }
