# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

from dataclasses import dataclass
import cudaq
import numpy


def test_tuple_assign_struq():

    @cudaq.kernel
    def test():
        q, r, s = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
        x(q, s)
        swap.ctrl(q, r, s)

    print(test)

    @cudaq.kernel
    def test1():
        q, r, s = cudaq.qvector(1), cudaq.qubit(), cudaq.qubit()
        x(q)
        x.ctrl(q, r)
        x.ctrl(q, s)

    print(test1)
    print("result test1: " + str(cudaq.sample(test1)))

    @cudaq.kernel
    def test2():
        q = cudaq.qvector(2), cudaq.qubit(), cudaq.qubit()
        x(q[0])
        x.ctrl(q[0], q[1])
        x.ctrl(q[0], q[2])

    print(test2)
    print("result test2: " + str(cudaq.sample(test2)))

    @cudaq.kernel
    def test3():
        q = cudaq.qvector(3), cudaq.qubit(), cudaq.qubit()
        c, t1, t2 = q
        x(c)
        x.ctrl(c, t1)
        x.ctrl(c, t2)

    print(test3)
    print("result test3: " + str(cudaq.sample(test3)))

    @cudaq.kernel
    def test4():
        c, q = cudaq.qvector(4), (cudaq.qubit(), cudaq.qubit())
        x(c)
        x.ctrl(c, q[0])
        x.ctrl(c, q[1])

    print(test4)
    print("result test4: " + str(cudaq.sample(test4)))

    @cudaq.kernel
    def test5():
        c, (q1, q2) = cudaq.qvector(5), (cudaq.qubit(), cudaq.qubit())
        x(c)
        x.ctrl(c, q1)
        x.ctrl(c, q2)

    print(test5)
    print("result test5: " + str(cudaq.sample(test5)))

    @cudaq.kernel
    def test6():
        q, a = cudaq.qubit(), numpy.pi
        ry(a, q)

    print(test6)
    print("result test6: " + str(cudaq.sample(test6)))

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.swap {{\[}}%[[VAL_0]]] %[[VAL_1]], %[[VAL_2]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   result test1: { 111:1000 }
# CHECK-LABEL:   result test2: { 1111:1000 }
# CHECK-LABEL:   result test3: { 11111:1000 }
# CHECK-LABEL:   result test4: { 111111:1000 }
# CHECK-LABEL:   result test5: { 1111111:1000 }
# CHECK-LABEL:   result test6: { 1:1000 }

def test_tuple_assign_struct():

    @cudaq.kernel
    def test1() -> float:
        q, r = 1, 2.
        return q + r

    print(test1)
    print("result test1: " + str(test1()))

    @cudaq.kernel
    def test2() -> float:
        v = 2, 2.
        return v[0] + v[1]

    print(test2)
    print("result test2: " + str(test2()))

    @cudaq.kernel
    def test3() -> float:
        v = 3, 2.
        v1, v2 = v
        return v1 + v2

    print(test3)
    print("result test3: " + str(test3()))

    # Use of tuples is not supported in kernels
    # @cudaq.kernel
    # def getTuple() -> tuple[int, float]:
    #     return 1, 2.

    @dataclass(slots=True)
    class MyTuple:
        first: int
        second: float

    @cudaq.kernel
    def getTuple(arg: int) -> MyTuple:
        return MyTuple(arg, 2.)

    @cudaq.kernel
    def test4() -> float:
        v = getTuple(4)
        return v[0] + v[1]

    print(test4)
    print("result test4: " + str(test4()))

    @cudaq.kernel
    def test5() -> float:
        v1, v2 = getTuple(5)
        return v1 + v2

    print(test5)
    print("result test5: " + str(test5()))

    @cudaq.kernel
    def test6() -> float:
        v = (getTuple(5), 1) 
        return v[0][0] + v[0][1] + v[1]

    print(test6)
    print("result test6: " + str(test6()))

    @cudaq.kernel
    def test7() -> float:
        v1, v2 = ((6, 2.), 1) 
        return v1[0] + v1[1] + v2

    print(test7)
    print("result test7: " + str(test7()))

    @cudaq.kernel
    def test8() -> float:
        (v1, v2), v3 = ((7, 2.), 1) 
        return v1 + v2 + v3

    print(test8)
    print("result test8: " + str(test8()))

    @cudaq.kernel
    def test9() -> float:
        (v1, v2), v3 = (getTuple(8), 1) 
        return v1 + v2 + v3

    print(test9)
    print("result test9: " + str(test9()))

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test1() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2.000000e+00 : f64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_2:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<i64>
# CHECK:           %[[VAL_3:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<f64>
# CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i64>
# CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_3]] : !cc.ptr<f64>
# CHECK:           %[[VAL_6:.*]] = cc.cast signed %[[VAL_4]] : (i64) -> f64
# CHECK:           %[[VAL_7:.*]] = arith.addf %[[VAL_6]], %[[VAL_5]] : f64
# CHECK:           return %[[VAL_7]] : f64
# CHECK:         result test1: 3.0

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test2() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f64
# CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.struct<"tuple" {i64, f64}>
# CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.struct<"tuple" {i64, f64}>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i64>
# CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][1] : (!cc.ptr<!cc.struct<"tuple" {i64, f64}>>) -> !cc.ptr<f64>
# CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<f64>
# CHECK:         result test2: 4.0

# CHECK-LABEL:   result test3: 5.0
# CHECK-LABEL:   result test4: 6.0
# CHECK-LABEL:   result test5: 7.0
# CHECK-LABEL:   result test6: 8.0
# CHECK-LABEL:   result test7: 9.0
# CHECK-LABEL:   result test8: 10.0
# CHECK-LABEL:   result test9: 11.0

def test_tuple_assign_failures():

    @cudaq.kernel
    def test1() -> float:
        v1, v2, v3 = ((1, 2), 3) 
        return v1 + v2 + v3

    try:
        print(test1)
    except Exception as e:
        print("Failure for test1:")
        print(e)

    @cudaq.kernel
    def test2():
        q1, q2, q3 = ((cudaq.qubit(), cudaq.qubit()), cudaq.qubit()) 
        x(q1)
        x.ctrl(q1, q2)
        x.ctrl(q1, q3)

    try:
        print(test2)
    except Exception as e:
        print("Failure for test2:")
        print(e)

    @cudaq.kernel
    def test3():
        v = cudaq.qubit(), 0.5
        rz(v[1], v[0])

    try:
        print(test3)
    except Exception as e:
        print("Failure for test3:")
        print(e)


# CHECK-LABEL:   Failure for test1:
# CHECK:         shape mismatch in tuple deconstruction
# CHECK-NEXT:    (offending source -> (v1, v2, v3) = ((1, 2), 3))

# CHECK-LABEL:   Failure for test2:
# CHECK:         shape mismatch in tuple deconstruction
# CHECK-NEXT:    (offending source -> (q1, q2, q3) = ((cudaq.qubit(), cudaq.qubit()), cudaq.qubit()))

# CHECK-LABEL:   Failure for test3:
# CHECK:         hybrid quantum-classical data types are not allowed
# CHECK-NEXT:    (offending source -> (cudaq.qubit(), 0.5))

if __name__ == '__main__':
    test_tuple_assign_struq()