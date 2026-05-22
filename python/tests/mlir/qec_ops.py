# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os
import pytest
import cudaq

# ---------------------------------------------------------------------------
# `cudaq.detector(h)` — single scalar handle.
# ---------------------------------------------------------------------------


def test_detector_scalar():

    @cudaq.kernel
    def kernel_detector_scalar():
        q = cudaq.qubit()
        h = mz(q)
        cudaq.detector(h)

    print(kernel_detector_scalar)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_detector_scalar
# CHECK-SAME:      attributes {"cudaq-entrypoint"
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "h" : (!quake.ref) -> !cc.measure_handle
# CHECK:           qec.detector %[[VAL_1]] : !cc.measure_handle
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# `cudaq.detector(h0, h1, h2)` — variadic scalar handles.
# ---------------------------------------------------------------------------


def test_detector_variadic():

    @cudaq.kernel
    def kernel_detector_variadic():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        q2 = cudaq.qubit()
        h0 = mz(q0)
        h1 = mz(q1)
        h2 = mz(q2)
        cudaq.detector(h0, h1, h2)

    print(kernel_detector_variadic)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_detector_variadic
# CHECK:           %[[H0:.*]] = quake.mz %{{.*}} name "h0" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[H1:.*]] = quake.mz %{{.*}} name "h1" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[H2:.*]] = quake.mz %{{.*}} name "h2" : (!quake.ref) -> !cc.measure_handle
# CHECK:           qec.detector %[[H0]], %[[H1]], %[[H2]] : !cc.measure_handle, !cc.measure_handle, !cc.measure_handle
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# `cudaq.detector(vec)` — single stdvec of handles.
# ---------------------------------------------------------------------------


def test_detector_vector():

    @cudaq.kernel
    def kernel_detector_vector():
        qs = cudaq.qvector(4)
        handles = mz(qs)
        cudaq.detector(handles)

    print(kernel_detector_vector)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_detector_vector
# CHECK:           %[[VS:.*]] = quake.mz %{{.*}} name "handles" : (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.detector %[[VS]] : !cc.stdvec<!cc.measure_handle>
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# `cudaq.logical_observable(...)` — variadic, default index 0 omitted.
# ---------------------------------------------------------------------------


def test_observable_variadic():

    @cudaq.kernel
    def kernel_observable_variadic():
        q0 = cudaq.qubit()
        q1 = cudaq.qubit()
        h0 = mz(q0)
        h1 = mz(q1)
        cudaq.logical_observable(h0, h1)

    print(kernel_observable_variadic)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_observable_variadic
# CHECK:           %[[H0:.*]] = quake.mz %{{.*}} name "h0"
# CHECK:           %[[H1:.*]] = quake.mz %{{.*}} name "h1"
# CHECK:           qec.observable %[[H0]], %[[H1]] : !cc.measure_handle, !cc.measure_handle
# CHECK-NOT:       index
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# `cudaq.logical_observable(vec)` — vector form, default index 0 omitted.
# ---------------------------------------------------------------------------


def test_observable_vector():

    @cudaq.kernel
    def kernel_observable_vector_default():
        qs = cudaq.qvector(3)
        handles = mz(qs)
        cudaq.logical_observable(handles)

    print(kernel_observable_vector_default)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_observable_vector_default
# CHECK:           %[[VS:.*]] = quake.mz %{{.*}} name "handles" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.observable %[[VS]] : !cc.stdvec<!cc.measure_handle>
# CHECK-NOT:       index
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# `cudaq.logical_observable(vec, observable_index=2)` — explicit index.
# ---------------------------------------------------------------------------


def test_observable_indexed():

    @cudaq.kernel
    def kernel_observable_explicit_idx():
        qs = cudaq.qvector(3)
        handles = mz(qs)
        cudaq.logical_observable(handles, observable_index=2)

    print(kernel_observable_explicit_idx)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_observable_explicit_idx
# CHECK:           %[[VS:.*]] = quake.mz %{{.*}} name "handles" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.observable %[[VS]] index 2 : !cc.stdvec<!cc.measure_handle>
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# `cudaq.detector(mz(q))` — nested call (no intermediate handle binding).
# ---------------------------------------------------------------------------


def test_detector_nested_mz():

    @cudaq.kernel
    def kernel_detector_nested_mz():
        q = cudaq.qubit()
        cudaq.detector(mz(q))

    print(kernel_detector_nested_mz)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_detector_nested_mz
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[H:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !cc.measure_handle
# CHECK:           qec.detector %[[H]] : !cc.measure_handle
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# Mixed shape: scalar handles + handle lists in the same call. The dialect
# op accepts any combination (`Variadic<AnyTypeOf<[scalar, list]>>`) and
# Q3's QIR conversion (`packMeasurementHandles` mixed-or-multi-stdvec
# branch) flattens the mix into a single `Result**` array, so a natural
# QEC-source-code call like "combine these stabilizers (list) with the
# boundary readout (scalar)" lowers cleanly.
# ---------------------------------------------------------------------------


def test_detector_mixed():

    @cudaq.kernel
    def kernel_detector_mixed():
        q = cudaq.qubit()
        qs = cudaq.qvector(2)
        h = mz(q)
        hs = mz(qs)
        cudaq.detector(hs, h)

    print(kernel_detector_mixed)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_detector_mixed
# CHECK:           %[[VAL_M:.*]] = quake.mz %{{.*}} name "h" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_HS:.*]] = quake.mz %{{.*}} name "hs" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.detector %[[VAL_HS]], %[[VAL_M]] : !cc.stdvec<!cc.measure_handle>, !cc.measure_handle
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# `cudaq.detectors(prev, curr)` — paired stdvecs.
# ---------------------------------------------------------------------------


def test_pair_detectors():

    @cudaq.kernel
    def kernel_pair_detectors():
        ancA = cudaq.qvector(3)
        ancB = cudaq.qvector(3)
        prev = mz(ancA)
        curr = mz(ancB)
        cudaq.detectors(prev, curr)

    print(kernel_pair_detectors)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_pair_detectors
# CHECK:           %[[P:.*]] = quake.mz %{{.*}} name "prev" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           %[[C:.*]] = quake.mz %{{.*}} name "curr" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.pair_detectors %[[P]], %[[C]] : <!cc.measure_handle>, <!cc.measure_handle>
# CHECK:           return
# CHECK:         }

# ---------------------------------------------------------------------------
# End-to-end distance-3 repetition code
# ---------------------------------------------------------------------------


def test_rep_code_d3():

    @cudaq.kernel
    def kernel_rep_code_d3(n_rounds: int):
        data = cudaq.qvector(3)
        anc0 = cudaq.qubit()
        anc1 = cudaq.qubit()
        prev_s0 = cudaq.measure_handle()
        prev_s1 = cudaq.measure_handle()

        for r in range(n_rounds):
            x.ctrl(data[0], anc0)
            x.ctrl(data[1], anc0)
            x.ctrl(data[1], anc1)
            x.ctrl(data[2], anc1)
            s0 = mz(anc0)
            s1 = mz(anc1)
            reset(anc0)
            reset(anc1)
            if r > 0:
                cudaq.detector(prev_s0, s0)
                cudaq.detector(prev_s1, s1)
            prev_s0 = s0
            prev_s1 = s1
        readout = mz(data)
        cudaq.logical_observable(readout)

    print(kernel_rep_code_d3)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_rep_code_d3
# CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 0 : i64
# CHECK:           %[[UNDEF_0:.*]] = cc.undef !cc.measure_handle
# CHECK:           %[[UNDEF_1:.*]] = cc.undef !cc.measure_handle
# CHECK:           %[[UNDEF_2:.*]] = cc.undef i64
# CHECK:           %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[ALLOCA_1:.*]] = quake.alloca !quake.ref
# CHECK:           %[[ALLOCA_2:.*]] = quake.alloca !quake.ref
# CHECK:           %[[UNDEF_3:.*]] = cc.undef !cc.measure_handle
# CHECK:           %[[UNDEF_4:.*]] = cc.undef !cc.measure_handle
# CHECK:           %[[LOOP_0:.*]]:6 = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_1]], %[[VAL_1:.*]] = %[[UNDEF_2]], %[[VAL_2:.*]] = %[[UNDEF_1]], %[[VAL_3:.*]] = %[[UNDEF_0]], %[[VAL_4:.*]] = %[[UNDEF_3]], %[[VAL_5:.*]] = %[[UNDEF_4]]) -> (i64, i64, !cc.measure_handle, !cc.measure_handle, !cc.measure_handle, !cc.measure_handle)) {
# CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[ARG0]] : i64
# CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : i64, i64, !cc.measure_handle, !cc.measure_handle, !cc.measure_handle, !cc.measure_handle)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_6:.*]]: i64, %[[VAL_7:.*]]: i64, %[[VAL_8:.*]]: !cc.measure_handle, %[[VAL_9:.*]]: !cc.measure_handle, %[[VAL_10:.*]]: !cc.measure_handle, %[[VAL_11:.*]]: !cc.measure_handle):
# CHECK:             %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]][0] : (!quake.veq<3>) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[EXTRACT_REF_0]]] %[[ALLOCA_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ALLOCA_0]][1] : (!quake.veq<3>) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[EXTRACT_REF_1]]] %[[ALLOCA_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             %[[EXTRACT_REF_2:.*]] = quake.extract_ref %[[ALLOCA_0]][1] : (!quake.veq<3>) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[EXTRACT_REF_2]]] %[[ALLOCA_2]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             %[[EXTRACT_REF_3:.*]] = quake.extract_ref %[[ALLOCA_0]][2] : (!quake.veq<3>) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[EXTRACT_REF_3]]] %[[ALLOCA_2]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             %[[MZ_0:.*]] = quake.mz %[[ALLOCA_1]] name "s0" : (!quake.ref) -> !cc.measure_handle
# CHECK:             %[[MZ_1:.*]] = quake.mz %[[ALLOCA_2]] name "s1" : (!quake.ref) -> !cc.measure_handle
# CHECK:             quake.reset %[[ALLOCA_1]] : (!quake.ref) -> ()
# CHECK:             quake.reset %[[ALLOCA_2]] : (!quake.ref) -> ()
# CHECK:             %[[CMPI_1:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[CONSTANT_1]] : i64
# CHECK:             cc.if(%[[CMPI_1]]) {
# CHECK:               qec.detector %[[VAL_10]], %[[MZ_0]] : !cc.measure_handle, !cc.measure_handle
# CHECK:               qec.detector %[[VAL_11]], %[[MZ_1]] : !cc.measure_handle, !cc.measure_handle
# CHECK:             } else {
# CHECK:             }
# CHECK:             cc.continue %[[VAL_6]], %[[VAL_6]], %[[MZ_0]], %[[MZ_1]], %[[MZ_0]], %[[MZ_1]] : i64, i64, !cc.measure_handle, !cc.measure_handle, !cc.measure_handle, !cc.measure_handle
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_12:.*]]: i64, %[[VAL_13:.*]]: i64, %[[VAL_14:.*]]: !cc.measure_handle, %[[VAL_15:.*]]: !cc.measure_handle, %[[VAL_16:.*]]: !cc.measure_handle, %[[VAL_17:.*]]: !cc.measure_handle):
# CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_12]], %[[CONSTANT_0]] : i64
# CHECK:             cc.continue %[[ADDI_0]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_17]] : i64, i64, !cc.measure_handle, !cc.measure_handle, !cc.measure_handle, !cc.measure_handle
# CHECK:           }
# CHECK:           %[[MZ_2:.*]] = quake.mz %[[ALLOCA_0]] name "readout" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.observable %[[MZ_2]] : !cc.stdvec<!cc.measure_handle>
# CHECK-NOT:       quake.discriminate
# CHECK:           return
# CHECK:         }

# ===========================================================================
# `cudaq.make_kernel()` programmatic builder surface.
# ===========================================================================


def test_b_detector_scalar():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc()
    h = kernel.mz(q)
    kernel.detector(h)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK:           %[[H:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           qec.detector %[[H]] : !cc.measure_handle
# CHECK:           return


def test_b_detector_vector():
    kernel = cudaq.make_kernel()
    qs = kernel.qalloc(4)
    hs = kernel.mz(qs)
    kernel.detector(hs)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK:           %[[HS:.*]] = quake.mz %{{.*}} : (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.detector %[[HS]] : !cc.stdvec<!cc.measure_handle>
# CHECK:           return


def test_b_detector_mixed():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc()
    qs = kernel.qalloc(2)
    h = kernel.mz(q)
    hs = kernel.mz(qs)
    kernel.detector(hs, h)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK:           %[[H:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[HS:.*]] = quake.mz %{{.*}} : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.detector %[[HS]], %[[H]] : !cc.stdvec<!cc.measure_handle>, !cc.measure_handle
# CHECK:           return


def test_b_observable_variadic():
    kernel = cudaq.make_kernel()
    q0 = kernel.qalloc()
    q1 = kernel.qalloc()
    h0 = kernel.mz(q0)
    h1 = kernel.mz(q1)
    kernel.logical_observable(h0, h1)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK:           %[[H0:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[H1:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
# CHECK:           qec.observable %[[H0]], %[[H1]] : !cc.measure_handle, !cc.measure_handle
# CHECK-NOT:       index
# CHECK:           return


def test_b_observable_vector():
    kernel = cudaq.make_kernel()
    qs = kernel.qalloc(3)
    hs = kernel.mz(qs)
    kernel.logical_observable(hs)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK:           %[[HS:.*]] = quake.mz %{{.*}} : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.observable %[[HS]] : !cc.stdvec<!cc.measure_handle>
# CHECK-NOT:       index
# CHECK:           return


def test_b_observable_indexed():
    kernel = cudaq.make_kernel()
    qs = kernel.qalloc(3)
    hs = kernel.mz(qs)
    kernel.logical_observable(hs, observable_index=2)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK:           %[[HS:.*]] = quake.mz %{{.*}} : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.observable %[[HS]] index 2 : !cc.stdvec<!cc.measure_handle>
# CHECK:           return


def test_b_pair_detectors():
    kernel = cudaq.make_kernel()
    qsA = kernel.qalloc(3)
    qsB = kernel.qalloc(3)
    prev = kernel.mz(qsA)
    curr = kernel.mz(qsB)
    kernel.detectors(prev, curr)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK:           %[[P:.*]] = quake.mz %{{.*}} : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           %[[C:.*]] = quake.mz %{{.*}} : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:           qec.pair_detectors %[[P]], %[[C]] : <!cc.measure_handle>, <!cc.measure_handle>
# CHECK:           return
