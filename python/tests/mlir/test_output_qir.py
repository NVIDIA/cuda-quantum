# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s



import cudaq


def test_synth_and_qir():

    @cudaq.kernel
    def ghz(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubitIdx in enumerate(range(numQubits - 1)):
            x.ctrl(qubits[i], qubits[qubitIdx + 1])

    print(cudaq.to_qir(ghz))
    ghz_synth = cudaq.synthesize(ghz, 5)
    print(cudaq.to_qir(ghz_synth, profile='qir-base'))


# CHECK:    %[[VAL_0:.*]] = tail call
# CHECK:    %[[VAL_1:.*]]* @__quantum__rt__qubit_allocate_array(i64
# CHECK:    %[[VAL_2:.*]])
# CHECK:         %[[VAL_3:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%[[VAL_1]]* %[[VAL_0]], i64 0)
# CHECK:         %[[VAL_4:.*]] = bitcast i8* %[[VAL_3]] to %[[VAL_5:.*]]**
# CHECK:         %[[VAL_6:.*]] = load %[[VAL_5]]*, %[[VAL_5]]** %[[VAL_4]], align 8
# CHECK:         tail call void @__quantum__qis__h(%[[VAL_5]]* %[[VAL_6]])
# CHECK:         %[[VAL_7:.*]] = add i64 %[[VAL_2]], -1
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
# CHECK:         br i1 %[[VAL_17]], label %[[VAL_11]], label %[[VAL_12]]
# CHECK:       ._crit_edge:                                      ; preds = %[[VAL_11]], %[[VAL_13]]
# CHECK:         %[[VAL_18:.*]] = alloca { i64, i64 }, i64 %[[VAL_8]], align 8
# CHECK:         %[[VAL_19:.*]] = icmp sgt i64 %[[VAL_8]], 0
# CHECK:         br i1 %[[VAL_19]], label %[[VAL_20:.*]], label %[[VAL_21:.*]]
# CHECK:       .preheader:                                       ; preds = %[[VAL_20]]
# CHECK:         br i1 %[[VAL_19]], label %[[VAL_22:.*]], label %[[VAL_21]]
# CHECK:       .lr.ph9:                                          ; preds = %[[VAL_12]], %[[VAL_20]]
# CHECK:         %[[VAL_23:.*]] = phi i64 [ %[[VAL_24:.*]], %[[VAL_20]] ], [ 0, %[[VAL_12]] ]
# CHECK:         %[[VAL_25:.*]] = getelementptr i64, i64* %[[VAL_9]], i64 %[[VAL_23]]
# CHECK:         %[[VAL_26:.*]] = load i64, i64* %[[VAL_25]], align 8
# CHECK:         %[[VAL_27:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_18]], i64 %[[VAL_23]], i32 0
# CHECK:         store i64 %[[VAL_23]], i64* %[[VAL_27]], align 8
# CHECK:         %[[VAL_28:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_18]], i64 %[[VAL_23]], i32 1
# CHECK:         store i64 %[[VAL_26]], i64* %[[VAL_28]], align 8
# CHECK:         %[[VAL_24]] = add nuw nsw i64 %[[VAL_23]], 1
# CHECK:         %[[VAL_29:.*]] = icmp slt i64 %[[VAL_24]], %[[VAL_8]]
# CHECK:         br i1 %[[VAL_29]], label %[[VAL_20]], label %[[VAL_30:.*]]
# CHECK:       .lr.ph10:                                         ; preds = %[[VAL_30]], %[[VAL_22]]
# CHECK:         %[[VAL_31:.*]] = phi i64 [ %[[VAL_32:.*]], %[[VAL_22]] ], [ 0, %[[VAL_30]] ]
# CHECK:         %[[VAL_33:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_18]], i64 %[[VAL_31]], i32 0
# CHECK:         %[[VAL_34:.*]] = load i64, i64* %[[VAL_33]], align 8
# CHECK:         %[[VAL_35:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_18]], i64 %[[VAL_31]], i32 1
# CHECK:         %[[VAL_36:.*]] = load i64, i64* %[[VAL_35]], align 8
# CHECK:         %[[VAL_37:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%[[VAL_1]]* %[[VAL_0]], i64 %[[VAL_34]])
# CHECK:         %[[VAL_38:.*]] = bitcast i8* %[[VAL_37]] to %[[VAL_5]]**
# CHECK:         %[[VAL_39:.*]] = load %[[VAL_5]]*, %[[VAL_5]]** %[[VAL_38]], align 8
# CHECK:         %[[VAL_40:.*]] = add i64 %[[VAL_36]], 1
# CHECK:         %[[VAL_41:.*]] = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%[[VAL_1]]* %[[VAL_0]], i64 %[[VAL_40]])
# CHECK:         %[[VAL_42:.*]] = bitcast i8* %[[VAL_41]] to %[[VAL_5]]**
# CHECK:         %[[VAL_43:.*]] = load %[[VAL_5]]*, %[[VAL_5]]** %[[VAL_42]], align 8
# CHECK:         tail call void (i64, void (%[[VAL_1]]*, %[[VAL_5]]*)*, ...) @invokeWithControlQubits(i64 1, void (%[[VAL_1]]*, %[[VAL_5]]*)* nonnull @__quantum__qis__x__ctl, %[[VAL_5]]* %[[VAL_39]], %[[VAL_5]]* %[[VAL_43]])
# CHECK:         %[[VAL_32]] = add nuw nsw i64 %[[VAL_31]], 1
# CHECK:         %[[VAL_44:.*]] = icmp slt i64 %[[VAL_32]], %[[VAL_8]]
# CHECK:         br i1 %[[VAL_44]], label %[[VAL_22]], label %[[VAL_21]]
# CHECK:       ._crit_edge11:                                    ; preds = %[[VAL_22]], %[[VAL_12]], %[[VAL_30]]
# CHECK:         tail call void @__quantum__rt__qubit_release_array(%[[VAL_1]]* %[[VAL_0]])
# CHECK:         ret void

# CHECK:   tail call void @__quantum__qis__h__body(
# CHECK:                                            %[[VAL_0:.*]]* null)
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* null, %[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 3 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 3 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 4 to %[[VAL_0]]*))
# CHECK:         ret void
