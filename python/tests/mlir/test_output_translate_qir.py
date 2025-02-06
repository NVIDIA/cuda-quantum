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

    print(cudaq.translate(ghz, format="qir"))
    ghz_synth = cudaq.synthesize(ghz, 5)
    print(cudaq.translate(ghz_synth, format='qir-base'))



# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz(i64 
# CHECK-SAME:                       %[[VAL_0:.*]]) local_unnamed_addr {
# CHECK:         %[[VAL_1:.*]] = tail call %[[VAL_2:.*]]* @__quantum__rt__qubit_allocate_array(i64 %[[VAL_0]])
# CHECK:         %[[VAL_3:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 0)
# CHECK:         %[[VAL_4:.*]] = bitcast i8* %[[VAL_3]] to %[[VAL_5:.*]]**
# CHECK:         %[[VAL_6:.*]] = load %[[VAL_5]]*, %[[VAL_5]]** %[[VAL_4]], align 8
# CHECK:         tail call void @__quantum__qis__h(%[[VAL_5]]* %[[VAL_6]])
# CHECK:         %[[VAL_7:.*]] = add i64 %[[VAL_0]], -1
# CHECK:         %[[VAL_8:.*]] = tail call i64 @llvm.abs.i64(i64 %[[VAL_7]], i1 false)
# CHECK:         %[[VAL_9:.*]] = alloca i64, i64 %[[VAL_8]], align 8
# CHECK:         %[[VAL_10:.*]] = icmp sgt i64 %[[VAL_7]], 0
# CHECK:         br i1 %[[VAL_10]], label %[[VAL_11:.*]], label %[[VAL_12:.*]]
# CHECK:       .lr.ph:                                           ; preds = %[[VAL_13:.*]], %[[VAL_11]]
# CHECK:         %[[VAL_14:.*]] = phi i64 [ %[[VAL_15:.*]], %[[VAL_11]] ], [ 0, %[[VAL_13]] ]
# CHECK:         %[[VAL_16:.*]] = getelementptr i64, i64* %[[VAL_9]], i64 %[[VAL_14]]
# CHECK:         store i64 %[[VAL_14]], i64* %[[VAL_16]], align 8
# CHECK:         %[[VAL_15]] = add nuw nsw i64 %[[VAL_14]], 1
# CHECK:         %[[VAL_17:.*]] = icmp slt i64 %[[VAL_15]], %[[VAL_7]]
# CHECK:         br i1 %[[VAL_17]], label %[[VAL_11]], label %[[VAL_18:.*]]
# CHECK:       :                                      ; preds = %[[VAL_11]]
# CHECK:         %[[VAL_19:.*]] = alloca { i64, i64 }, i64 %[[VAL_7]], align 8
# CHECK:         br i1 %[[VAL_10]], label %[[VAL_20:.*]], label %[[VAL_12]]
# CHECK:       :                                       ; preds = %[[VAL_20]]
# CHECK:         br i1 %[[VAL_10]], label %[[VAL_21:.*]], label %[[VAL_12]]
# CHECK:       .lr.ph10:                                         ; preds = %[[VAL_18]], %[[VAL_20]]
# CHECK:         %[[VAL_22:.*]] = phi i64 [ %[[VAL_23:.*]], %[[VAL_20]] ], [ 0, %[[VAL_18]] ]
# CHECK:         %[[VAL_24:.*]] = getelementptr i64, i64* %[[VAL_9]], i64 %[[VAL_22]]
# CHECK:         %[[VAL_25:.*]] = load i64, i64* %[[VAL_24]], align 8
# CHECK:         %[[VAL_26:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_19]], i64 %[[VAL_22]], i32 0
# CHECK:         store i64 %[[VAL_22]], i64* %[[VAL_26]], align 8
# CHECK:         %[[VAL_27:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_19]], i64 %[[VAL_22]], i32 1
# CHECK:         store i64 %[[VAL_25]], i64* %[[VAL_27]], align 8
# CHECK:         %[[VAL_23]] = add nuw nsw i64 %[[VAL_22]], 1
# CHECK:         %[[VAL_28:.*]] = icmp slt i64 %[[VAL_23]], %[[VAL_7]]
# CHECK:         br i1 %[[VAL_28]], label %[[VAL_20]], label %[[VAL_29:.*]]
# CHECK:       :                                         ; preds = %[[VAL_29]], %[[VAL_21]]
# CHECK:         %[[VAL_30:.*]] = phi i64 [ %[[VAL_31:.*]], %[[VAL_21]] ], [ 0, %[[VAL_29]] ]
# CHECK:         %[[VAL_32:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_19]], i64 %[[VAL_30]], i32 0
# CHECK:         %[[VAL_33:.*]] = load i64, i64* %[[VAL_32]], align 8
# CHECK:         %[[VAL_34:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_19]], i64 %[[VAL_30]], i32 1
# CHECK:         %[[VAL_35:.*]] = load i64, i64* %[[VAL_34]], align 8
# CHECK:         %[[VAL_36:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 %[[VAL_33]])
# CHECK:         %[[VAL_37:.*]] = bitcast i8* %[[VAL_36]] to %[[VAL_5]]**
# CHECK:         %[[VAL_38:.*]] = load %[[VAL_5]]*, %[[VAL_5]]** %[[VAL_37]], align 8
# CHECK:         %[[VAL_39:.*]] = add i64 %[[VAL_35]], 1
# CHECK:         %[[VAL_40:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 %[[VAL_39]])
# CHECK:         %[[VAL_41:.*]] = bitcast i8* %[[VAL_40]] to %[[VAL_5]]**
# CHECK:         %[[VAL_42:.*]] = load %[[VAL_5]]*, %[[VAL_5]]** %[[VAL_41]], align 8
# CHECK:         tail call void (i64, void (%[[VAL_2]]*, %[[VAL_5]]*)*, ...) @invokeWithControlQubits(i64 1, void (%[[VAL_2]]*, %[[VAL_5]]*)* nonnull @__quantum__qis__x__ctl, %[[VAL_5]]* %[[VAL_38]], %[[VAL_5]]* %[[VAL_42]])
# CHECK:         %[[VAL_31]] = add nuw nsw i64 %[[VAL_30]], 1
# CHECK:         %[[VAL_43:.*]] = icmp slt i64 %[[VAL_31]], %[[VAL_7]]
# CHECK:         br i1 %[[VAL_43]], label %[[VAL_21]], label %[[VAL_12]]
# CHECK:       :                                    ; preds = %[[VAL_21]], %[[VAL_13]], %[[VAL_18]], %[[VAL_29]]
# CHECK:         tail call void @__quantum__rt__qubit_release_array(%[[VAL_2]]* %[[VAL_1]])
# CHECK:         ret void
# CHECK:       }

# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz()
# CHECK:         tail call void @__quantum__qis__h__body(%[[VAL_0:.*]]* null)
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* null, %[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 3 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 3 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 4 to %[[VAL_0]]*))
# CHECK:         ret void
# CHECK:       }

