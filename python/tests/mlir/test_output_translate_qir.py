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
# CHECK:         %[[VAL_1:.*]] = call %[[VAL_2:.*]]* @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         %[[VAL_3:.*]] = call %[[VAL_4:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 0)
# CHECK:         %[[VAL_5:.*]] = load %[[VAL_4]]*, %[[VAL_4]]** %[[VAL_3]], align 8
# CHECK:         call void @__quantum__qis__h(%[[VAL_4]]* %[[VAL_5]])
# CHECK:         %[[VAL_6:.*]] = alloca [2 x i64], align 8
# CHECK:         %[[VAL_7:.*]] = bitcast [2 x i64]* %[[VAL_6]] to i64*
# CHECK:         store i64 0, i64* %[[VAL_7]], align 8
# CHECK:         %[[VAL_8:.*]] = getelementptr [2 x i64], [2 x i64]* %[[VAL_6]], i32 0, i32 1
# CHECK:         store i64 1, i64* %[[VAL_8]], align 8
# CHECK:         %[[VAL_9:.*]] = load i64, i64* %[[VAL_7]], align 8
# CHECK:         %[[VAL_10:.*]] = add i64 %[[VAL_9]], 1
# CHECK:         %[[VAL_11:.*]] = call %[[VAL_4]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 %[[VAL_10]])
# CHECK:         %[[VAL_12:.*]] = load %[[VAL_4]]*, %[[VAL_4]]** %[[VAL_11]], align 8
# CHECK:         %[[VAL_13:.*]] = bitcast %[[VAL_4]]* %[[VAL_5]] to i8*
# CHECK:         %[[VAL_14:.*]] = bitcast %[[VAL_4]]* %[[VAL_12]] to i8*
# CHECK:         call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* bitcast (void (%[[VAL_2]]*, %[[VAL_4]]*)* @__quantum__qis__x__ctl to i8*), i8* %[[VAL_13]], i8* %[[VAL_14]])
# CHECK:         %[[VAL_15:.*]] = load i64, i64* %[[VAL_8]], align 8
# CHECK:         %[[VAL_16:.*]] = call %[[VAL_4]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 1)
# CHECK:         %[[VAL_17:.*]] = load %[[VAL_4]]*, %[[VAL_4]]** %[[VAL_16]], align 8
# CHECK:         %[[VAL_18:.*]] = add i64 %[[VAL_15]], 1
# CHECK:         %[[VAL_19:.*]] = call %[[VAL_4]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 %[[VAL_18]])
# CHECK:         %[[VAL_20:.*]] = load %[[VAL_4]]*, %[[VAL_4]]** %[[VAL_19]], align 8
# CHECK:         %[[VAL_21:.*]] = bitcast %[[VAL_4]]* %[[VAL_17]] to i8*
# CHECK:         %[[VAL_22:.*]] = bitcast %[[VAL_4]]* %[[VAL_20]] to i8*
# CHECK:         call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* bitcast (void (%[[VAL_2]]*, %[[VAL_4]]*)* @__quantum__qis__x__ctl to i8*), i8* %[[VAL_21]], i8* %[[VAL_22]])
# CHECK:         call void @__quantum__rt__qubit_release_array(%[[VAL_2]]* %[[VAL_1]])
# CHECK:         ret void
# CHECK:       }

# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz
# CHECK-SAME:    (i64 %[[VAL_0:.*]])
# CHECK:         call void @__quantum__qis__h__body(%[[VAL_1:.*]]* null)
# CHECK:         %[[VAL_2:.*]] = alloca [4 x i64], align 8
# CHECK:         %[[VAL_3:.*]] = bitcast [4 x i64]* %[[VAL_2]] to i64*
# CHECK:         store i64 0, i64* %[[VAL_3]], align 8
# CHECK:         %[[VAL_4:.*]] = getelementptr [4 x i64], [4 x i64]* %[[VAL_2]], i32 0, i32 1
# CHECK:         store i64 1, i64* %[[VAL_4]], align 8
# CHECK:         %[[VAL_5:.*]] = getelementptr [4 x i64], [4 x i64]* %[[VAL_2]], i32 0, i32 2
# CHECK:         store i64 2, i64* %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = getelementptr [4 x i64], [4 x i64]* %[[VAL_2]], i32 0, i32 3
# CHECK:         store i64 3, i64* %[[VAL_6]], align 8
# CHECK:         %[[VAL_7:.*]] = load i64, i64* %[[VAL_3]], align 8
# CHECK:         %[[VAL_8:.*]] = add i64 %[[VAL_7]], 1
# CHECK:         %[[VAL_9:.*]] = getelementptr [5 x i64], [5 x i64]* @__nvqpp__mlirgen__ghz..{{.*}}.rodata_0, i32 0, i64 %[[VAL_8]]
# CHECK:         %[[VAL_10:.*]] = load i64, i64* %[[VAL_9]], align 8
# CHECK:         %[[VAL_11:.*]] = inttoptr i64 %[[VAL_10]] to %[[VAL_1]]*
# CHECK:         call void @__quantum__qis__cnot__body(%[[VAL_1]]* null, %[[VAL_1]]* %[[VAL_11]])
# CHECK:         %[[VAL_12:.*]] = load i64, i64* %[[VAL_4]], align 8
# CHECK:         %[[VAL_13:.*]] = add i64 %[[VAL_12]], 1
# CHECK:         %[[VAL_14:.*]] = getelementptr [5 x i64], [5 x i64]* @__nvqpp__mlirgen__ghz..{{.*}}.rodata_0, i32 0, i64 %[[VAL_13]]
# CHECK:         %[[VAL_15:.*]] = load i64, i64* %[[VAL_14]], align 8
# CHECK:         %[[VAL_16:.*]] = inttoptr i64 %[[VAL_15]] to %[[VAL_1]]*
# CHECK:         call void @__quantum__qis__cnot__body(%[[VAL_1]]* inttoptr (i64 1 to %[[VAL_1]]*), %[[VAL_1]]* %[[VAL_16]])
# CHECK:         %[[VAL_17:.*]] = load i64, i64* %[[VAL_5]], align 8
# CHECK:         %[[VAL_18:.*]] = add i64 %[[VAL_17]], 1
# CHECK:         %[[VAL_19:.*]] = getelementptr [5 x i64], [5 x i64]* @__nvqpp__mlirgen__ghz..{{.*}}.rodata_0, i32 0, i64 %[[VAL_18]]
# CHECK:         %[[VAL_20:.*]] = load i64, i64* %[[VAL_19]], align 8
# CHECK:         %[[VAL_21:.*]] = inttoptr i64 %[[VAL_20]] to %[[VAL_1]]*
# CHECK:         call void @__quantum__qis__cnot__body(%[[VAL_1]]* inttoptr (i64 2 to %[[VAL_1]]*), %[[VAL_1]]* %[[VAL_21]])
# CHECK:         %[[VAL_22:.*]] = load i64, i64* %[[VAL_6]], align 8
# CHECK:         %[[VAL_23:.*]] = add i64 %[[VAL_22]], 1
# CHECK:         %[[VAL_24:.*]] = getelementptr [5 x i64], [5 x i64]* @__nvqpp__mlirgen__ghz..{{.*}}.rodata_0, i32 0, i64 %[[VAL_23]]
# CHECK:         %[[VAL_25:.*]] = load i64, i64* %[[VAL_24]], align 8
# CHECK:         %[[VAL_26:.*]] = inttoptr i64 %[[VAL_25]] to %[[VAL_1]]*
# CHECK:         call void @__quantum__qis__cnot__body(%[[VAL_1]]* inttoptr (i64 3 to %[[VAL_1]]*), %[[VAL_1]]* %[[VAL_26]])
# CHECK:         ret void
# CHECK:       }
