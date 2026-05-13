# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_synth_and_translate():

    @cudaq.kernel
    def ghz(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubitIdx in enumerate(range(numQubits - 1)):
            x.ctrl(qubits[i], qubits[qubitIdx + 1])

    print(cudaq.translate(ghz, 3, format="qir"))
    ghz_synth = cudaq.synthesize(ghz, 5)
    print(cudaq.translate(ghz_synth, format='qir-base'))


# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz
# CHECK-SAME:    (i64 %[[VAL_0:.*]]) {
# CHECK:         %[[VAL_2:.*]] = call ptr @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         %[[VAL_3:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_2]], i64 0)
# CHECK:         %[[VAL_4:.*]] = load ptr, ptr %[[VAL_3]], align 8
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_4]])
# CHECK:         %[[VAL_5:.*]] = alloca [2 x i64], align 8
# CHECK:         store i64 0, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = getelementptr [2 x i64], ptr %[[VAL_5]], i32 0, i32 1
# CHECK:         store i64 1, ptr %[[VAL_6]], align 8
# CHECK:         %[[VAL_7:.*]] = load i64, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_8:.*]] = add i64 %[[VAL_7]], 1
# CHECK:         %[[VAL_9:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_2]], i64 %[[VAL_8]])
# CHECK:         %[[VAL_10:.*]] = load ptr, ptr %[[VAL_9]], align 8
# CHECK:         call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr @__quantum__qis__x__ctl, ptr %[[VAL_4]], ptr %[[VAL_10]])
# CHECK:         %[[VAL_11:.*]] = load i64, ptr %[[VAL_6]], align 8
# CHECK:         %[[VAL_12:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_2]], i64 1)
# CHECK:         %[[VAL_13:.*]] = load ptr, ptr %[[VAL_12]], align 8
# CHECK:         %[[VAL_14:.*]] = add i64 %[[VAL_11]], 1
# CHECK:         %[[VAL_15:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_2]], i64 %[[VAL_14]])
# CHECK:         %[[VAL_16:.*]] = load ptr, ptr %[[VAL_15]], align 8
# CHECK:         call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr @__quantum__qis__x__ctl, ptr %[[VAL_13]], ptr %[[VAL_16]])
# CHECK:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_2]])
# CHECK:         ret void
# CHECK:       }

# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz
# CHECK-SAME:    (i64 %[[VAL_0:.*]]) #0 {
# CHECK:         call void @__quantum__qis__h__body(ptr null)
# CHECK:         %[[VAL_2:.*]] = alloca [4 x i64], align 8
# CHECK:         store i64 0, ptr %[[VAL_2]], align 8
# CHECK:         %[[VAL_3:.*]] = getelementptr [4 x i64], ptr %[[VAL_2]], i32 0, i32 1
# CHECK:         store i64 1, ptr %[[VAL_3]], align 8
# CHECK:         %[[VAL_4:.*]] = getelementptr [4 x i64], ptr %[[VAL_2]], i32 0, i32 2
# CHECK:         store i64 2, ptr %[[VAL_4]], align 8
# CHECK:         %[[VAL_5:.*]] = getelementptr [4 x i64], ptr %[[VAL_2]], i32 0, i32 3
# CHECK:         store i64 3, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = load i64, ptr %[[VAL_2]], align 8
# CHECK:         %[[VAL_7:.*]] = add i64 %[[VAL_6]], 1
# CHECK:         %[[VAL_8:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz..{{.*}}.rodata_0, i32 0, i64 %[[VAL_7]]
# CHECK:         %[[VAL_9:.*]] = load i64, ptr %[[VAL_8]], align 8
# CHECK:         %[[VAL_10:.*]] = inttoptr i64 %[[VAL_9]] to ptr
# CHECK:         call void @__quantum__qis__cnot__body(ptr null, ptr %[[VAL_10]])
# CHECK:         %[[VAL_11:.*]] = load i64, ptr %[[VAL_3]], align 8
# CHECK:         %[[VAL_12:.*]] = add i64 %[[VAL_11]], 1
# CHECK:         %[[VAL_13:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz..{{.*}}.rodata_0, i32 0, i64 %[[VAL_12]]
# CHECK:         %[[VAL_14:.*]] = load i64, ptr %[[VAL_13]], align 8
# CHECK:         %[[VAL_15:.*]] = inttoptr i64 %[[VAL_14]] to ptr
# CHECK:         call void @__quantum__qis__cnot__body(ptr inttoptr (i64 1 to ptr), ptr %[[VAL_15]])
# CHECK:         %[[VAL_16:.*]] = load i64, ptr %[[VAL_4]], align 8
# CHECK:         %[[VAL_17:.*]] = add i64 %[[VAL_16]], 1
# CHECK:         %[[VAL_18:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz..{{.*}}.rodata_0, i32 0, i64 %[[VAL_17]]
# CHECK:         %[[VAL_19:.*]] = load i64, ptr %[[VAL_18]], align 8
# CHECK:         %[[VAL_20:.*]] = inttoptr i64 %[[VAL_19]] to ptr
# CHECK:         call void @__quantum__qis__cnot__body(ptr inttoptr (i64 2 to ptr), ptr %[[VAL_20]])
# CHECK:         %[[VAL_21:.*]] = load i64, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_22:.*]] = add i64 %[[VAL_21]], 1
# CHECK:         %[[VAL_23:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz..{{.*}}.rodata_0, i32 0, i64 %[[VAL_22]]
# CHECK:         %[[VAL_24:.*]] = load i64, ptr %[[VAL_23]], align 8
# CHECK:         %[[VAL_25:.*]] = inttoptr i64 %[[VAL_24]] to ptr
# CHECK:         call void @__quantum__qis__cnot__body(ptr inttoptr (i64 3 to ptr), ptr %[[VAL_25]])
# CHECK:         ret void
# CHECK:       }
