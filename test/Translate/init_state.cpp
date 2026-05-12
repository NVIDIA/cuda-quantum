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
// CHECK:         %[[VAL_0:.*]] = tail call ptr @__nvqpp_cudaq_state_createFromData_complex_f64(ptr nonnull @__nvqpp__mlirgen__kernel.rodata_0, i64 4)
// CHECK:         %[[VAL_1:.*]] = tail call i64 @__nvqpp_cudaq_state_numberOfQubits(ptr %[[VAL_0]])
// CHECK:         %[[VAL_2:.*]] = tail call ptr @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_1]], ptr %[[VAL_0]])
// CHECK:         tail call void @__nvqpp_cudaq_state_delete(ptr %[[VAL_0]])
// CHECK:         %[[VAL_4:.*]] = tail call i64 @__quantum__rt__array_get_size_1d(ptr %[[VAL_2]])
// CHECK:         %[[VAL_5:.*]] = icmp sgt i64 %[[VAL_4]], 0
// CHECK:         br i1 %[[VAL_5]], label %[[VAL_6:.*]], label %[[VAL_7:.*]]
// CHECK:         ; preds = %[[VAL_8:.*]], %[[VAL_6]]
// CHECK:         %[[VAL_9:.*]] = phi i64 [ %[[VAL_10:.*]], %[[VAL_6]] ], [ 0, %[[VAL_8]] ]
// CHECK:         %[[VAL_11:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_2]], i64 %[[VAL_9]])
// CHECK:         %[[VAL_13:.*]] = load ptr, ptr %[[VAL_11]], align 8
// CHECK:         %[[VAL_14:.*]] = tail call ptr @__quantum__qis__mz__to__register(ptr %[[VAL_13]], ptr nonnull @cstr.726573756C7400)
// CHECK:         %[[VAL_10]] = add nuw nsw i64 %[[VAL_9]], 1
// CHECK:         %[[VAL_16:.*]] = icmp eq i64 %[[VAL_10]], %[[VAL_4]]
// CHECK:         br i1 %[[VAL_16]], label %[[VAL_7]], label %[[VAL_6]]
// CHECK:         ; preds = %[[VAL_6]], %[[VAL_8]]
// CHECK:         tail call void @__quantum__rt__qubit_release_array(ptr %[[VAL_2]])
// CHECK:         ret void
// clang-format on
