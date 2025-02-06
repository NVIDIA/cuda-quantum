# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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

    print(cudaq.translate(ghz, format="qir"))
    ghz_synth = cudaq.synthesize(ghz, 5)
    print(cudaq.translate(ghz_synth, format='qir-base'))


# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz(i64 
# CHECK-SAME:                              %[[VAL_0:.*]]) local_unnamed_addr {
# CHECK:         %[[VAL_1:.*]] = tail call %[[VAL_2:.*]]* @__quantum__rt__qubit_allocate_array(i64 %[[VAL_0]])
# CHECK:         %[[VAL_3:.*]] = tail call %[[VAL_4:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 0)
# CHECK:         %[[VAL_5:.*]] = load %[[VAL_4]]*, %[[VAL_4]]** %[[VAL_3]]
# CHECK:         tail call void @__quantum__qis__h(%[[VAL_4]]* %[[VAL_5]])
# CHECK:         %[[VAL_6:.*]] = add i64 %[[VAL_0]], -1
# CHECK:         %[[VAL_7:.*]] = tail call i64 @llvm.abs.i64(i64 %[[VAL_6]], i1 false)
# CHECK:         %[[VAL_8:.*]] = alloca i64, i64 %[[VAL_7]]
# CHECK:         %[[VAL_9:.*]] = icmp sgt i64 %[[VAL_6]], 0
# CHECK:         br i1 %[[VAL_9]], label %[[VAL_10:.*]], label %[[VAL_11:.*]]
# CHECK:       :                                           ; preds = %[[VAL_12:.*]], %[[VAL_10]]
# CHECK:         %[[VAL_13:.*]] = phi i64 [ %[[VAL_14:.*]], %[[VAL_10]] ], [ 0, %[[VAL_12]] ]
# CHECK:         %[[VAL_15:.*]] = getelementptr i64, i64* %[[VAL_8]], i64 %[[VAL_13]]
# CHECK:         store i64 %[[VAL_13]], i64* %[[VAL_15]]
# CHECK:         %[[VAL_14]] = add nuw nsw i64 %[[VAL_13]], 1
# CHECK:         %[[VAL_16:.*]] = icmp slt i64 %[[VAL_14]], %[[VAL_6]]
# CHECK:         br i1 %[[VAL_16]], label %[[VAL_10]], label %[[VAL_17:.*]]
# CHECK:       :                                      ; preds = %[[VAL_10]]
# CHECK:         %[[VAL_18:.*]] = alloca { i64, i64 }, i64 %[[VAL_6]]
# CHECK:         br i1 %[[VAL_9]], label %[[VAL_19:.*]], label %[[VAL_11]]
# CHECK:       :                                       ; preds = %[[VAL_19]]
# CHECK:         br i1 %[[VAL_9]], label %[[VAL_20:.*]], label %[[VAL_11]]
# CHECK:       :                                         ; preds = %[[VAL_17]], %[[VAL_19]]
# CHECK:         %[[VAL_21:.*]] = phi i64 [ %[[VAL_22:.*]], %[[VAL_19]] ], [ 0, %[[VAL_17]] ]
# CHECK:         %[[VAL_23:.*]] = getelementptr i64, i64* %[[VAL_8]], i64 %[[VAL_21]]
# CHECK:         %[[VAL_24:.*]] = load i64, i64* %[[VAL_23]]
# CHECK:         %[[VAL_25:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_18]], i64 %[[VAL_21]], i32 0
# CHECK:         store i64 %[[VAL_21]], i64* %[[VAL_25]]
# CHECK:         %[[VAL_26:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_18]], i64 %[[VAL_21]], i32 1
# CHECK:         store i64 %[[VAL_24]], i64* %[[VAL_26]]
# CHECK:         %[[VAL_22]] = add nuw nsw i64 %[[VAL_21]], 1
# CHECK:         %[[VAL_27:.*]] = icmp slt i64 %[[VAL_22]], %[[VAL_6]]
# CHECK:         br i1 %[[VAL_27]], label %[[VAL_19]], label %[[VAL_28:.*]]
# CHECK:       :                                         ; preds = %[[VAL_28]], %[[VAL_20]]
# CHECK:         %[[VAL_29:.*]] = phi i64 [ %[[VAL_30:.*]], %[[VAL_20]] ], [ 0, %[[VAL_28]] ]
# CHECK:         %[[VAL_31:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_18]], i64 %[[VAL_29]], i32 0
# CHECK:         %[[VAL_32:.*]] = load i64, i64* %[[VAL_31]]
# CHECK:         %[[VAL_33:.*]] = getelementptr { i64, i64 }, { i64, i64 }* %[[VAL_18]], i64 %[[VAL_29]], i32 1
# CHECK:         %[[VAL_34:.*]] = load i64, i64* %[[VAL_33]]
# CHECK:         %[[VAL_35:.*]] = tail call %[[VAL_4]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 %[[VAL_32]])
# CHECK:         %[[VAL_36:.*]] = bitcast %[[VAL_4]]** %[[VAL_35]] to i8**
# CHECK:         %[[VAL_37:.*]] = load i8*, i8** %[[VAL_36]]
# CHECK:         %[[VAL_38:.*]] = add i64 %[[VAL_34]], 1
# CHECK:         %[[VAL_39:.*]] = tail call %[[VAL_4]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 %[[VAL_38]])
# CHECK:         %[[VAL_40:.*]] = bitcast %[[VAL_4]]** %[[VAL_39]] to i8**
# CHECK:         %[[VAL_41:.*]] = load i8*, i8** %[[VAL_40]]
# CHECK:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%[[VAL_2]]*, %[[VAL_4]]*)* @__quantum__qis__x__ctl to i8*), i8* %[[VAL_37]], i8* %[[VAL_41]])
# CHECK:         %[[VAL_30]] = add nuw nsw i64 %[[VAL_29]], 1
# CHECK:         %[[VAL_42:.*]] = icmp slt i64 %[[VAL_30]], %[[VAL_6]]
# CHECK:         br i1 %[[VAL_42]], label %[[VAL_20]], label %[[VAL_11]]
# CHECK:       :                                    ; preds = %[[VAL_20]], %[[VAL_12]], %[[VAL_17]], %[[VAL_28]]
# CHECK:         tail call void @__quantum__rt__qubit_release_array(%[[VAL_2]]* %[[VAL_1]])
# CHECK:         ret void
# CHECK:       }

# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz() local_unnamed_addr #0 {
# CHECK:         tail call void @__quantum__qis__h__body(%[[VAL_0:.*]]* null)
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* null, %[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 3 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 3 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 4 to %[[VAL_0]]*))
# CHECK:         ret void
# CHECK:       }

