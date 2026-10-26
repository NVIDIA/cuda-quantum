# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP %s | FileCheck %s

import cudaq
from dataclasses import dataclass


def test_list_deconstruction():

    @dataclass(slots=True)
    class MyTuple:
        q: cudaq.qview
        r: cudaq.qubit

    @cudaq.kernel
    def kernel1():
        q0, q1, q2 = cudaq.qvector(3)
        x(q0)
        y(q1)
        z(q2)

    print(kernel1)

    @cudaq.kernel
    def kernel2():
        (q0, q1), q2 = cudaq.qvector(2), cudaq.qubit()
        x(q0)
        y(q1)
        z(q2)

    print(kernel2)

    @cudaq.kernel
    def kernel3():
        ts = cudaq.qvector(2)
        data = MyTuple(ts, cudaq.qubit())
        (q0, q1), q2 = data
        x(q0)
        y(q1)
        z(q2)

    print(kernel3)

    @cudaq.kernel
    def kernel4():
        (q0,), (q1, q2) = cudaq.qvector(1), cudaq.qvector(2)
        x(q0)
        y(q1)
        z(q2)

    print(kernel4)

    @cudaq.kernel
    def kernel5():
        r1, r2 = [0.5, 1.]
        q = cudaq.qubit()
        rz(r1, q)
        ry(r2, q)

    print(kernel5)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][2] : (!quake.veq<3>) -> !quake.ref
# CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.y %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.z %[[VAL_3]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
# CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.y %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.z %[[VAL_3]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.y %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.z %[[VAL_3]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel4
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<1>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<1>) -> !quake.ref
# CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_4]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.y %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.z %[[VAL_3]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel5
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 5.000000e-01 : f64
# CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
# CHECK:           quake.rz (%[[VAL_1]]) %[[VAL_4]] : (f64, !quake.ref) -> ()
# CHECK:           quake.ry (%[[VAL_0]]) %[[VAL_4]] : (f64, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }


def test_list_deconstruction_failures():

    @dataclass(slots=True)
    class MyTuple:
        q: cudaq.qview
        r: cudaq.qubit

    try:

        @cudaq.kernel
        def kernel1():
            (q0, q1), q2 = cudaq.qvector(3)
            x(q0)
            y(q1)
            z(q2)

        print(kernel1)
    except Exception as e:
        print("Failure for kernel1:")
        print(e)

    try:

        @cudaq.kernel
        def kernel2():
            ts = cudaq.qvector(1)
            data = MyTuple(ts, cudaq.qubit())
            (q0, q1), q2 = data
            x(q0)
            y(q1)
            z(q2)

        # This will fail when running the pass manager.
        # I could not figure out how to capture any errors that the
        # pass manager produces and return them along with the
        # exception. I think it would be nicer to do that, since
        # as it is, the error message is a bit fragmented.
        print(kernel2)
    except Exception as e:
        print("Failure for kernel2:")
        print(e)

    # Note: We have a slight mismatch in Python interpreted code
    # vs compiled python code; since the length of the list we are
    # deconstructing is runtime information, we merely index into
    # it to get however many elements we assign to. If this number
    # is larger than the length of the list, then we either get a
    # pass manager failure or a runtime failure, depending on when
    # the length of the list is known. However, if the number `n`
    # of targets we assign to is smaller than the list, we happily
    # will just assign the first n items without a failure. This
    # is in contrast to the Python interpreter that doesn't allow
    # for that. We can, and do, give a proper error, however, if
    # the list we assign is a Python literal expression

    try:

        @cudaq.kernel
        def kernel3():
            r1, r2 = [0.5, 1., 1.5]
            q = cudaq.qubit()
            rz(r1, q)
            ry(r2, q)

        print(kernel3)
    except Exception as e:
        print("Failure for kernel3:")
        print(e)


# CHECK-LABEL:   Failure for kernel1:
# CHECK:         shape mismatch in tuple deconstruction
# CHECK-NEXT:    (offending source -> {{.*}}q0, q1{{.*}}, q2{{.*}} = cudaq.qvector(3))

# CHECK-LABEL:   Failure for kernel2:
# CHECK-NEXT:    could not compile code for 'kernel2

# CHECK-LABEL:   Failure for kernel3:
# CHECK:         shape mismatch in tuple deconstruction
# CHECK-NEXT:    (offending source -> {{.*}}r1, r2{{.*}} = [0.5, 1.0, 1.5])
