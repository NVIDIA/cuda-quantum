# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
# CHECK:         %[[VAL_1:.*]] = call ptr @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         %[[VAL_2:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_1]], i64 0)
# CHECK:         %[[VAL_3:.*]] = load ptr, ptr %[[VAL_2]], align 8
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_3]])
# CHECK:         %[[VAL_4:.*]] = alloca [2 x i64], align 8
# CHECK:         store i64 0, ptr %[[VAL_4]], align 8
# CHECK:         %[[VAL_5:.*]] = getelementptr [2 x i64], ptr %[[VAL_4]], i32 0, i32 1
# CHECK:         store i64 1, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = alloca [2 x i64], align 8
# CHECK:         store i64 0, ptr %[[VAL_6]], align 8
# CHECK:         %[[VAL_7:.*]] = getelementptr [2 x i64], ptr %[[VAL_6]], i32 0, i32 1
# CHECK:         store i64 1, ptr %[[VAL_7]], align 8
# CHECK:         %[[VAL_8:.*]] = alloca [2 x { i64, i64 }], align 8
# CHECK:         %[[VAL_9:.*]] = load i64, ptr %[[VAL_6]], align 8
# CHECK:         %[[VAL_10:.*]] = insertvalue { i64, i64 } { i64 0, i64 undef }, i64 %[[VAL_9]], 1
# CHECK:         store { i64, i64 } %[[VAL_10]], ptr %[[VAL_8]], align 8
# CHECK:         %[[VAL_11:.*]] = load i64, ptr %[[VAL_7]], align 8
# CHECK:         %[[VAL_12:.*]] = getelementptr [2 x { i64, i64 }], ptr %[[VAL_8]], i32 0, i32 1
# CHECK:         %[[VAL_13:.*]] = insertvalue { i64, i64 } { i64 1, i64 undef }, i64 %[[VAL_11]], 1
# CHECK:         store { i64, i64 } %[[VAL_13]], ptr %[[VAL_12]], align 8
# CHECK:         %[[VAL_14:.*]] = load { i64, i64 }, ptr %[[VAL_8]], align 8
# CHECK:         %[[VAL_15:.*]] = extractvalue { i64, i64 } %[[VAL_14]], 0
# CHECK:         %[[VAL_16:.*]] = extractvalue { i64, i64 } %[[VAL_14]], 1
# CHECK:         %[[VAL_17:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_1]], i64 %[[VAL_15]])
# CHECK:         %[[VAL_18:.*]] = load ptr, ptr %[[VAL_17]], align 8
# CHECK:         %[[VAL_19:.*]] = add i64 %[[VAL_16]], 1
# CHECK:         %[[VAL_20:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_1]], i64 %[[VAL_19]])
# CHECK:         %[[VAL_21:.*]] = load ptr, ptr %[[VAL_20]], align 8
# CHECK:         call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr @__quantum__qis__x__ctl, ptr %[[VAL_18]], ptr %[[VAL_21]])
# CHECK:         %[[VAL_22:.*]] = load { i64, i64 }, ptr %[[VAL_12]], align 8
# CHECK:         %[[VAL_23:.*]] = extractvalue { i64, i64 } %[[VAL_22]], 0
# CHECK:         %[[VAL_24:.*]] = extractvalue { i64, i64 } %[[VAL_22]], 1
# CHECK:         %[[VAL_25:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_1]], i64 %[[VAL_23]])
# CHECK:         %[[VAL_26:.*]] = load ptr, ptr %[[VAL_25]], align 8
# CHECK:         %[[VAL_27:.*]] = add i64 %[[VAL_24]], 1
# CHECK:         %[[VAL_28:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_1]], i64 %[[VAL_27]])
# CHECK:         %[[VAL_29:.*]] = load ptr, ptr %[[VAL_28]], align 8
# CHECK:         call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr @__quantum__qis__x__ctl, ptr %[[VAL_26]], ptr %[[VAL_29]])
# CHECK:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_1]])
# CHECK:         ret void
# CHECK:       }

# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz
# CHECK:         call void @__quantum__qis__h__body(ptr null)
# CHECK:         %[[VAL_0:.*]] = alloca [4 x i64], align 8
# CHECK:         store i64 0, ptr %[[VAL_0]], align 8
# CHECK:         %[[VAL_1:.*]] = getelementptr [4 x i64], ptr %[[VAL_0]], i32 0, i32 1
# CHECK:         store i64 1, ptr %[[VAL_1]], align 8
# CHECK:         %[[VAL_2:.*]] = getelementptr [4 x i64], ptr %[[VAL_0]], i32 0, i32 2
# CHECK:         store i64 2, ptr %[[VAL_2]], align 8
# CHECK:         %[[VAL_3:.*]] = getelementptr [4 x i64], ptr %[[VAL_0]], i32 0, i32 3
# CHECK:         store i64 3, ptr %[[VAL_3]], align 8
# CHECK:         %[[VAL_4:.*]] = alloca [4 x i64], align 8
# CHECK:         store i64 0, ptr %[[VAL_4]], align 8
# CHECK:         %[[VAL_5:.*]] = getelementptr [4 x i64], ptr %[[VAL_4]], i32 0, i32 1
# CHECK:         store i64 1, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = getelementptr [4 x i64], ptr %[[VAL_4]], i32 0, i32 2
# CHECK:         store i64 2, ptr %[[VAL_6]], align 8
# CHECK:         %[[VAL_7:.*]] = getelementptr [4 x i64], ptr %[[VAL_4]], i32 0, i32 3
# CHECK:         store i64 3, ptr %[[VAL_7]], align 8
# CHECK:         %[[VAL_8:.*]] = alloca [4 x { i64, i64 }], align 8
# CHECK:         %[[VAL_9:.*]] = load i64, ptr %[[VAL_4]], align 8
# CHECK:         %[[VAL_10:.*]] = insertvalue { i64, i64 } { i64 0, i64 undef }, i64 %[[VAL_9]], 1
# CHECK:         store { i64, i64 } %[[VAL_10]], ptr %[[VAL_8]], align 8
# CHECK:         %[[VAL_11:.*]] = load i64, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_12:.*]] = getelementptr [4 x { i64, i64 }], ptr %[[VAL_8]], i32 0, i32 1
# CHECK:         %[[VAL_13:.*]] = insertvalue { i64, i64 } { i64 1, i64 undef }, i64 %[[VAL_11]], 1
# CHECK:         store { i64, i64 } %[[VAL_13]], ptr %[[VAL_12]], align 8
# CHECK:         %[[VAL_14:.*]] = load i64, ptr %[[VAL_6]], align 8
# CHECK:         %[[VAL_15:.*]] = getelementptr [4 x { i64, i64 }], ptr %[[VAL_8]], i32 0, i32 2
# CHECK:         %[[VAL_16:.*]] = insertvalue { i64, i64 } { i64 2, i64 undef }, i64 %[[VAL_14]], 1
# CHECK:         store { i64, i64 } %[[VAL_16]], ptr %[[VAL_15]], align 8
# CHECK:         %[[VAL_17:.*]] = load i64, ptr %[[VAL_7]], align 8
# CHECK:         %[[VAL_18:.*]] = getelementptr [4 x { i64, i64 }], ptr %[[VAL_8]], i32 0, i32 3
# CHECK:         %[[VAL_19:.*]] = insertvalue { i64, i64 } { i64 3, i64 undef }, i64 %[[VAL_17]], 1
# CHECK:         store { i64, i64 } %[[VAL_19]], ptr %[[VAL_18]], align 8
# CHECK:         %[[VAL_20:.*]] = load { i64, i64 }, ptr %[[VAL_8]], align 8
# CHECK:         %[[VAL_21:.*]] = extractvalue { i64, i64 } %[[VAL_20]], 0
# CHECK:         %[[VAL_22:.*]] = extractvalue { i64, i64 } %[[VAL_20]], 1
# CHECK:         %[[VAL_23:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz{{.*}}.rodata_0, i32 0, i64 %[[VAL_21]]
# CHECK:         %[[VAL_24:.*]] = load i64, ptr %[[VAL_23]], align 8
# CHECK:         %[[VAL_25:.*]] = inttoptr i64 %[[VAL_24]] to ptr
# CHECK:         %[[VAL_26:.*]] = add i64 %[[VAL_22]], 1
# CHECK:         %[[VAL_27:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz{{.*}}.rodata_0, i32 0, i64 %[[VAL_26]]
# CHECK:         %[[VAL_28:.*]] = load i64, ptr %[[VAL_27]], align 8
# CHECK:         %[[VAL_29:.*]] = inttoptr i64 %[[VAL_28]] to ptr
# CHECK:         call void @__quantum__qis__cnot__body(ptr %[[VAL_25]], ptr %[[VAL_29]])
# CHECK:         %[[VAL_30:.*]] = load { i64, i64 }, ptr %[[VAL_12]], align 8
# CHECK:         %[[VAL_31:.*]] = extractvalue { i64, i64 } %[[VAL_30]], 0
# CHECK:         %[[VAL_32:.*]] = extractvalue { i64, i64 } %[[VAL_30]], 1
# CHECK:         %[[VAL_33:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz{{.*}}.rodata_0, i32 0, i64 %[[VAL_31]]
# CHECK:         %[[VAL_34:.*]] = load i64, ptr %[[VAL_33]], align 8
# CHECK:         %[[VAL_35:.*]] = inttoptr i64 %[[VAL_34]] to ptr
# CHECK:         %[[VAL_36:.*]] = add i64 %[[VAL_32]], 1
# CHECK:         %[[VAL_37:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz{{.*}}.rodata_0, i32 0, i64 %[[VAL_36]]
# CHECK:         %[[VAL_38:.*]] = load i64, ptr %[[VAL_37]], align 8
# CHECK:         %[[VAL_39:.*]] = inttoptr i64 %[[VAL_38]] to ptr
# CHECK:         call void @__quantum__qis__cnot__body(ptr %[[VAL_35]], ptr %[[VAL_39]])
# CHECK:         %[[VAL_40:.*]] = load { i64, i64 }, ptr %[[VAL_15]], align 8
# CHECK:         %[[VAL_41:.*]] = extractvalue { i64, i64 } %[[VAL_40]], 0
# CHECK:         %[[VAL_42:.*]] = extractvalue { i64, i64 } %[[VAL_40]], 1
# CHECK:         %[[VAL_43:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz{{.*}}.rodata_0, i32 0, i64 %[[VAL_41]]
# CHECK:         %[[VAL_44:.*]] = load i64, ptr %[[VAL_43]], align 8
# CHECK:         %[[VAL_45:.*]] = inttoptr i64 %[[VAL_44]] to ptr
# CHECK:         %[[VAL_46:.*]] = add i64 %[[VAL_42]], 1
# CHECK:         %[[VAL_47:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz{{.*}}.rodata_0, i32 0, i64 %[[VAL_46]]
# CHECK:         %[[VAL_48:.*]] = load i64, ptr %[[VAL_47]], align 8
# CHECK:         %[[VAL_49:.*]] = inttoptr i64 %[[VAL_48]] to ptr
# CHECK:         call void @__quantum__qis__cnot__body(ptr %[[VAL_45]], ptr %[[VAL_49]])
# CHECK:         %[[VAL_50:.*]] = load { i64, i64 }, ptr %[[VAL_18]], align 8
# CHECK:         %[[VAL_51:.*]] = extractvalue { i64, i64 } %[[VAL_50]], 0
# CHECK:         %[[VAL_52:.*]] = extractvalue { i64, i64 } %[[VAL_50]], 1
# CHECK:         %[[VAL_53:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz{{.*}}.rodata_0, i32 0, i64 %[[VAL_51]]
# CHECK:         %[[VAL_54:.*]] = load i64, ptr %[[VAL_53]], align 8
# CHECK:         %[[VAL_55:.*]] = inttoptr i64 %[[VAL_54]] to ptr
# CHECK:         %[[VAL_56:.*]] = add i64 %[[VAL_52]], 1
# CHECK:         %[[VAL_57:.*]] = getelementptr [5 x i64], ptr @__nvqpp__mlirgen__ghz{{.*}}.rodata_0, i32 0, i64 %[[VAL_56]]
# CHECK:         %[[VAL_58:.*]] = load i64, ptr %[[VAL_57]], align 8
# CHECK:         %[[VAL_59:.*]] = inttoptr i64 %[[VAL_58]] to ptr
# CHECK:         call void @__quantum__qis__cnot__body(ptr %[[VAL_55]], ptr %[[VAL_59]])
# CHECK:         ret void
# CHECK:       }
