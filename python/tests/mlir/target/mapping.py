# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: CUDAQ_DUMP_JIT_IR=1 PYTHONPATH=../../.. python3 %s --target oqc --emulate 2>&1 | FileCheck %s

import cudaq
from cudaq import spin


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


@cudaq.kernel
def state_prep():
    q0, q1, q2 = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
    x(q0)
    x(q1)
    x.ctrl(q0, q1)
    x.ctrl(q0, q2)


observe_result = cudaq.observe(state_prep, spin.z(0))

print('expectation {:.1f}'.format(observe_result.expectation()))

# CHECK-LABEL:   define void @__nvqpp__mlirgen__foo
# CHECK:         tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 1 to ptr))
# CHECK:         tail call void @__quantum__qis__x__body(ptr null)
# CHECK:         tail call void @__quantum__qis__cnot__body(ptr nonnull inttoptr (i64 1 to ptr), ptr null)
# CHECK-NOT:     tail call void @__quantum__qis__swap__body
# CHECK:         tail call void @__quantum__qis__cnot__body(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull inttoptr (i64 2 to ptr))
# CHECK:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 1 to ptr), ptr writeonly null)
# CHECK:         tail call void @__quantum__qis__mz__body(ptr null, ptr nonnull writeonly inttoptr (i64 1 to ptr))
# CHECK:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull writeonly inttoptr (i64 2 to ptr))
# CHECK:         tail call void @__quantum__rt__array_record_output(i64 3, ptr nonnull @cstr.{{.*}})
# CHECK:         tail call void @__quantum__rt__result_record_output(ptr nonnull null, ptr nonnull @cstr.{{.*}})
# CHECK:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull @cstr.{{.*}})
# CHECK:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull @cstr.{{.*}})
# CHECK:         ret void
# CHECK-LABEL:   define void @__nvqpp__mlirgen__state_prep
# CHECK:         tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 1 to ptr))
# CHECK:         tail call void @__quantum__qis__x__body(ptr null)
# CHECK:         tail call void @__quantum__qis__cnot__body(ptr nonnull inttoptr (i64 1 to ptr), ptr null)
# CHECK-NOT:     tail call void @__quantum__qis__swap__body
# CHECK:         tail call void @__quantum__qis__cnot__body(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull inttoptr (i64 2 to ptr))
# CHECK:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 1 to ptr), ptr writeonly null)
# CHECK:         ret void
# CHECK:         most_probable "101"
# CHECK:         expectation -1.0
