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
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_1]])
# CHECK:         call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr @__quantum__qis__x__ctl, ptr %[[VAL_1]], ptr %[[VAL_2]])
# CHECK:         call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr @__quantum__qis__x__ctl, ptr %[[VAL_2]], ptr %[[VAL_3]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_1]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_2]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_3]])
# CHECK:         ret void
# CHECK:       }

# CHECK-LABEL: define void @__nvqpp__mlirgen__ghz..0x
# CHECK-SAME:      %[[VAL_0:.*]])
# CHECK:         call void @__quantum__qis__h__body(ptr null)
# CHECK:         call void @__quantum__qis__cnot__body(ptr null, ptr inttoptr (i64 1 to ptr))
# CHECK:         call void @__quantum__qis__cnot__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 2 to ptr))
# CHECK:         call void @__quantum__qis__cnot__body(ptr inttoptr (i64 2 to ptr), ptr inttoptr (i64 3 to ptr))
# CHECK:         call void @__quantum__qis__cnot__body(ptr inttoptr (i64 3 to ptr), ptr inttoptr (i64 4 to ptr))
# CHECK:         ret void
# CHECK:       }
