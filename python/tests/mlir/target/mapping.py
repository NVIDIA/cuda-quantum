# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: CUDAQ_DUMP_JIT_IR=1 PYTHONPATH=../../.. python3 %s --target oqc --emulate |& FileCheck %s

import cudaq


@cudaq.kernel
def foo():
    q0, q1, q2 = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
    x(q0)
    x(q1)
    x.ctrl(q0, q1)
    x.ctrl(q0, q2)
    q0result = mz(q0)
    q1result = mz(q1)
    q2result = mz(q2)


result = cudaq.sample(foo)

print('most_probable "{}"'.format(result.most_probable()))

# CHECK:         tail call void @__quantum__qis__x__body(%[[VAL_0:.*]]* null)
# CHECK:         tail call void @__quantum__qis__x__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* null, %[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__swap__body(%[[VAL_0]]* null, %[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*))
# CHECK:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* writeonly null)
# CHECK:         tail call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull writeonly inttoptr (i64 1 to %Result*))
# CHECK:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull writeonly inttoptr (i64 2 to %Result*))
# CHECK:         tail call void @__quantum__rt__result_record_output(%Result* null, i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @cstr.{{.*}}, i64 0, i64 0))
# CHECK:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 1 to %Result*), i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @cstr.{{.*}}, i64 0, i64 0))
# CHECK:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 2 to %Result*), i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @cstr.{{.*}}, i64 0, i64 0))
# CHECK:         ret void
# CHECK:         most_probable "101"
