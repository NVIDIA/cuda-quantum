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


# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz..0x
# CHECK-SAME:      %[[VAL_0:.*]])
# CHECK:         %[[VAL_1:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_2:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_3:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_4:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_5:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_4]], i64 0)
# CHECK:         store ptr %[[VAL_1]], ptr %[[VAL_5]]
# CHECK:         %[[VAL_6:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_7:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_6]], i64 0)
# CHECK:         store ptr %[[VAL_2]], ptr %[[VAL_7]]
# CHECK:         %[[VAL_8:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_4]], ptr %[[VAL_6]])
# CHECK:         %[[VAL_9:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_10:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_9]], i64 0)
# CHECK:         store ptr %[[VAL_3]], ptr %[[VAL_10]]
# CHECK:         %[[VAL_11:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_8]], ptr %[[VAL_9]])
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_1]])
# CHECK:         %[[VAL_12:.*]] = alloca [2 x i64]
# CHECK:         store i64 0, ptr %[[VAL_12]]
# CHECK:         %[[VAL_13:.*]] = getelementptr [2 x i64], ptr %[[VAL_12]], i32 0, i32 1
# CHECK:         store i64 1, ptr %[[VAL_13]]
# CHECK:         %[[VAL_14:.*]] = load i64, ptr %[[VAL_12]]
# CHECK:         %[[VAL_15:.*]] = add i64 %[[VAL_14]], 1
# CHECK:         %[[VAL_16:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_11]], i64 %[[VAL_15]])
# CHECK:         %[[VAL_17:.*]] = load ptr, ptr %[[VAL_16]]
# CHECK:         call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr @__quantum__qis__x__ctl, ptr %[[VAL_1]], ptr %[[VAL_17]])
# CHECK:         %[[VAL_18:.*]] = load i64, ptr %[[VAL_13]]
# CHECK:         %[[VAL_19:.*]] = add i64 %[[VAL_18]], 1
# CHECK:         %[[VAL_20:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_11]], i64 %[[VAL_19]])
# CHECK:         %[[VAL_21:.*]] = load ptr, ptr %[[VAL_20]]
# CHECK:         call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr @__quantum__qis__x__ctl, ptr %[[VAL_2]], ptr %[[VAL_21]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_1]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_2]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_3]])
# CHECK:         ret void
# CHECK:       }

# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz..0x
# CHECK-SAME:      %[[VAL_0:.*]])
# CHECK:         %[[VAL_1:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_2:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_1]], i64 0)
# CHECK:         store ptr null, ptr %[[VAL_2]]
# CHECK:         %[[VAL_3:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_4:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_3]], i64 0)
# CHECK:         store ptr inttoptr (i64 1 to ptr), ptr %[[VAL_4]]
# CHECK:         %[[VAL_5:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_1]], ptr %[[VAL_3]])
# CHECK:         %[[VAL_6:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_7:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_6]], i64 0)
# CHECK:         store ptr inttoptr (i64 2 to ptr), ptr %[[VAL_7]]
# CHECK:         %[[VAL_8:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_5]], ptr %[[VAL_6]])
# CHECK:         %[[VAL_9:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_10:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_9]], i64 0)
# CHECK:         store ptr inttoptr (i64 3 to ptr), ptr %[[VAL_10]]
# CHECK:         %[[VAL_11:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_8]], ptr %[[VAL_9]])
# CHECK:         %[[VAL_12:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_13:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_12]], i64 0)
# CHECK:         store ptr inttoptr (i64 4 to ptr), ptr %[[VAL_13]]
# CHECK:         %[[VAL_14:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_11]], ptr %[[VAL_12]])
# CHECK:         call void @__quantum__qis__h__body(ptr null)
# CHECK:         %[[VAL_15:.*]] = alloca [4 x i64]
# CHECK:         store i64 0, ptr %[[VAL_15]]
# CHECK:         %[[VAL_16:.*]] = getelementptr [4 x i64], ptr %[[VAL_15]], i32 0, i32 1
# CHECK:         store i64 1, ptr %[[VAL_16]]
# CHECK:         %[[VAL_17:.*]] = getelementptr [4 x i64], ptr %[[VAL_15]], i32 0, i32 2
# CHECK:         store i64 2, ptr %[[VAL_17]]
# CHECK:         %[[VAL_18:.*]] = getelementptr [4 x i64], ptr %[[VAL_15]], i32 0, i32 3
# CHECK:         store i64 3, ptr %[[VAL_18]]
# CHECK:         %[[VAL_19:.*]] = load i64, ptr %[[VAL_15]]
# CHECK:         %[[VAL_20:.*]] = add i64 %[[VAL_19]], 1
# CHECK:         %[[VAL_21:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_14]], i64 %[[VAL_20]])
# CHECK:         %[[VAL_22:.*]] = load ptr, ptr %[[VAL_21]]
# CHECK:         call void @__quantum__qis__cnot__body(ptr null, ptr %[[VAL_22]])
# CHECK:         %[[VAL_23:.*]] = load i64, ptr %[[VAL_16]]
# CHECK:         %[[VAL_24:.*]] = add i64 %[[VAL_23]], 1
# CHECK:         %[[VAL_25:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_14]], i64 %[[VAL_24]])
# CHECK:         %[[VAL_26:.*]] = load ptr, ptr %[[VAL_25]]
# CHECK:         call void @__quantum__qis__cnot__body(ptr inttoptr (i64 1 to ptr), ptr %[[VAL_26]])
# CHECK:         %[[VAL_27:.*]] = load i64, ptr %[[VAL_17]]
# CHECK:         %[[VAL_28:.*]] = add i64 %[[VAL_27]], 1
# CHECK:         %[[VAL_29:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_14]], i64 %[[VAL_28]])
# CHECK:         %[[VAL_30:.*]] = load ptr, ptr %[[VAL_29]]
# CHECK:         call void @__quantum__qis__cnot__body(ptr inttoptr (i64 2 to ptr), ptr %[[VAL_30]])
# CHECK:         %[[VAL_31:.*]] = load i64, ptr %[[VAL_18]]
# CHECK:         %[[VAL_32:.*]] = add i64 %[[VAL_31]], 1
# CHECK:         %[[VAL_33:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_14]], i64 %[[VAL_32]])
# CHECK:         %[[VAL_34:.*]] = load ptr, ptr %[[VAL_33]]
# CHECK:         call void @__quantum__qis__cnot__body(ptr inttoptr (i64 3 to ptr), ptr %[[VAL_34]])
# CHECK:         ret void
# CHECK:       }
