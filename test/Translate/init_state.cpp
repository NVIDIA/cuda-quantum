/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
// CHECK:         %[[VAL_0:.*]] = tail call ptr @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 2, ptr nonnull @__nvqpp__mlirgen__kernel.rodata_0)
// CHECK:         %[[VAL_1:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 0)
// CHECK:         %[[VAL_2:.*]] = load ptr, ptr %[[VAL_1]], align 8
// CHECK:         %[[VAL_3:.*]] = tail call ptr @__quantum__qis__mz__to__register(ptr %[[VAL_2]], ptr nonnull @cstr.726573756C7400)
// CHECK:         %[[VAL_4:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 1)
// CHECK:         %[[VAL_5:.*]] = load ptr, ptr %[[VAL_4]], align 8
// CHECK:         %[[VAL_6:.*]] = tail call ptr @__quantum__qis__mz__to__register(ptr %[[VAL_5]], ptr nonnull @cstr.726573756C7400)
// CHECK:         tail call void @__quantum__rt__qubit_release_array(ptr %[[VAL_0]])
// CHECK:         ret void
// clang-format on
