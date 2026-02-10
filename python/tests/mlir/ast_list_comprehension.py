# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

from dataclasses import dataclass
import cudaq


def test_list_comprehension_constant():

    @cudaq.kernel
    def kernel1() -> bool:
        combined = [True for _ in range(5)]
        res = False
        for v in combined:
            res = res or v
        return res

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel1)

    @cudaq.kernel
    def kernel2() -> float:
        combined = [1.0 for _ in range(5)]
        res = 0.
        for v in combined:
            res += v
        return res

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 5.0)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)

    @cudaq.kernel
    def kernel3() -> float:
        combined = [1.j for _ in range(5)]
        res = 0.j
        for v in combined:
            res += v
        return res.imag

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == 5.0)
    # keep after assert, such that we have no output if assert fails
    print(kernel3)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1..
# CHECK-SAME: () -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_6:.*]] = cc.alloca !cc.array<i64 x 5>
# CHECK:           %[[VAL_7:.*]]:2 = cc.loop while ((%[[VAL_8:.*]] = %{{.*}}, %[[VAL_9:.*]] = %{{.*}}) -> (i64, i64)) {
# CHECK:           } do {
# CHECK:             %[[VAL_13:.*]] = cc.compute_ptr %{{.*}} : (!cc.ptr<!cc.array<i64 x 5>>, i64) -> !cc.ptr<i64>
# CHECK:             cc.store %{{.*}}, %[[VAL_13]] : !cc.ptr<i64>
# CHECK:           } step {
# CHECK:           %[[VAL_18:.*]] = cc.alloca !cc.array<i8 x 5>
# CHECK:           %[[VAL_19:.*]] = cc.loop while ((%[[VAL_20:.*]] = %{{.*}}) -> (i64)) {
# CHECK:           } do {
# CHECK:             %[[VAL_23:.*]] = cc.compute_ptr %{{.*}} : (!cc.ptr<!cc.array<i8 x 5>>, i64) -> !cc.ptr<i8>
# CHECK:             cc.store %{{.*}}, %[[VAL_23]] : !cc.ptr<i8>
# CHECK:           } step {
# CHECK:           %[[VAL_26:.*]]:3 = cc.loop while ((%[[VAL_27:.*]] =
# CHECK-SAME: ) -> (i64, i1, i1)) {
# CHECK:           } do {
# CHECK:             %[[VAL_32:.*]] = cc.compute_ptr %{{.*}} : (!cc.ptr<!cc.array<i8 x 5>>, i64) -> !cc.ptr<i8>
# CHECK:             %[[VAL_33:.*]] = cc.load %[[VAL_32]] : !cc.ptr<i8>
# CHECK:             %[[VAL_35:.*]] = cc.if(%[[VAL_31:.*]]) -> i1 {
# CHECK:               cc.continue %{{.*}} : i1
# CHECK:             } else {
# CHECK:               cc.continue %{{.*}} : i1
# CHECK:             cc.continue %{{.*}}, %[[VAL_36:.*]] : i64, i1
# CHECK:           } step {
# CHECK:           return %{{.*}} : i1

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1..
# CHECK-SAME: .run() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, quake.cudaq_run = [i1]} {

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2..
# CHECK-SAME () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_5:.*]] = cc.alloca !cc.array<i64 x 5>
# CHECK:           %[[VAL_6:.*]]:2 = cc.loop while ((%[[VAL_7:.*]] =
# CHECK-SAME: ) -> (i64, i64)) {
# CHECK:           } do {
# CHECK:             %[[VAL_12:.*]] = cc.compute_ptr %{{.*}} : (!cc.ptr<!cc.array<i64 x 5>>, i64) -> !cc.ptr<i64>
# CHECK:           } step {
# CHECK:           %[[VAL_17:.*]] = cc.alloca !cc.array<f64 x 5>
# CHECK:           %[[VAL_18:.*]] = cc.loop while ((%[[VAL_19:.*]] = %
# CHECK-SAME: ) -> (i64)) {
# CHECK:           } do {
# CHECK:             %[[VAL_22:.*]] = cc.compute_ptr %{{.*}} : (!cc.ptr<!cc.array<f64 x 5>>, i64) -> !cc.ptr<f64>
# CHECK:           } step {
# CHECK:           %[[VAL_25:.*]]:3 = cc.loop while ((%[[VAL_26:.*]] =
# CHECK-SAME: ) -> (i64, f64, f64)) {
# CHECK:           } do {
# CHECK:             %[[VAL_31:.*]] = cc.compute_ptr %
# CHECK-SAME: : (!cc.ptr<!cc.array<f64 x 5>>, i64) -> !cc.ptr<f64>
# CHECK:           } step {
# CHECK:           }
# CHECK:           return %{{.*}} : f64

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3..
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_5:.*]] = cc.alloca !cc.array<i64 x 5>
# CHECK:           %[[VAL_6:.*]]:2 = cc.loop while ((%[[VAL_7:.*]] =
# CHECK-SAME: ) -> (i64, i64)) {
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           }
# CHECK:           %[[VAL_18:.*]] = cc.loop while ((%[[VAL_19:.*]] =
# CHECK-SAME: ) -> (i64)) {
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           %[[VAL_25:.*]]:3 = cc.loop while ((%[[VAL_26:.*]] = %
# CHECK-SAME: ) -> (i64, complex<f64>, complex<f64>)) {
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           }
# CHECK:           return %{{.*}} : f64
# CHECK:         }


def test_list_comprehension_variable():

    @cudaq.kernel
    def kernel1() -> bool:
        c = True
        combined = [c for _ in range(5)]
        res = False
        for v in combined:
            res = res or v
        return res

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel1)

    @cudaq.kernel
    def kernel2() -> int:
        c = 1.0
        combined = [c for _ in range(5)]
        res = 0.
        for v in combined:
            res += v
        return int(res)

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 5)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)

    @cudaq.kernel
    def kernel3() -> float:
        c = 1.j
        combined = [c for _ in range(5)]
        res = 0.j
        for v in combined:
            res += v
        return res.imag

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == 5.0)
    # keep after assert, such that we have no output if assert fails
    print(kernel3)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1..
# CHECK-SAME: () -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:           %[[VAL_6:.*]] = cc.alloca !cc.array<i64 x 5>
# CHECK:           %[[VAL_7:.*]]:2 = cc.loop while ((%[[VAL_8:.*]] = %
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           }
# CHECK:           %[[VAL_18:.*]] = cc.alloca !cc.array<i8 x 5>
# CHECK:           %[[VAL_19:.*]] = cc.loop while ((%[[VAL_20:.*]] = %
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           %[[VAL_26:.*]]:3 = cc.loop while ((%[[VAL_27:.*]] = %
# CHECK:           } do {
# CHECK:             %[[VAL_35:.*]] = cc.if(%{{.*}}) -> i1 {
# CHECK:             } else {
# CHECK:             }
# CHECK:           } step {
# CHECK:           }
# CHECK:           return %{{.*}} : i1
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2..
# CHECK-SAME: () -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:           %[[VAL_5:.*]] = cc.alloca !cc.array<i64 x 5>
# CHECK:           %[[VAL_6:.*]]:2 = cc.loop while ((%[[VAL_7:.*]] =
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           }
# CHECK:           %[[VAL_17:.*]] = cc.alloca !cc.array<f64 x 5>
# CHECK:           %[[VAL_18:.*]] = cc.loop while ((%[[VAL_19:.*]] = %
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           %[[VAL_25:.*]]:3 = cc.loop while ((%[[VAL_26:.*]] = %
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           }
# CHECK:           return %{{.*}} : i64
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2..
# CHECK-SAME: .run() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, quake.cudaq_run = [i64]} {

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3..
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:           %[[VAL_5:.*]] = cc.alloca !cc.array<i64 x 5>
# CHECK:           %[[VAL_6:.*]]:2 = cc.loop while ((%[[VAL_7:.*]] = %
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           }
# CHECK:           %[[VAL_17:.*]] = cc.alloca !cc.array<complex<f64> x 5>
# CHECK:           %[[VAL_18:.*]] = cc.loop while ((%[[VAL_19:.*]] = %
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           %[[VAL_25:.*]]:3 = cc.loop while ((%[[VAL_26:.*]] = %
# CHECK:           } do {
# CHECK:           } step {
# CHECK:           }
# CHECK:           %[[VAL_37:.*]] = complex.im %[[VAL_38:.*]]#{{[0-9]}} : complex<f64>
# CHECK:           return %[[VAL_37]] : f64
# CHECK:         }


def test_list_comprehension_capture():

    c = True

    @cudaq.kernel
    def kernel1() -> bool:
        combined = [c for _ in range(5)]
        res = False
        for v in combined:
            res = res or v
        return res

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel1)

    c = 1.0

    @cudaq.kernel
    def kernel2() -> float:
        combined = [c for _ in range(5)]
        res = 0.
        for v in combined:
            res += v
        return res

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 5.)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)

    c = 1.j

    @cudaq.kernel
    def kernel3() -> float:
        combined = [c for _ in range(5)]
        res = 0.j
        for v in combined:
            res += v
        return res.imag

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == 5.)
    # keep after assert, such that we have no output if assert fails
    print(kernel3)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1..
# CHECK-SAME: (%[[VAL_0:.*]]: i1 {quake.pylifted}) -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2..
# CHECK-SAME: (%[[VAL_0:.*]]: f64 {quake.pylifted}) -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3..
# CHECK-SAME: (%[[VAL_0:.*]]: complex<f64> {quake.pylifted}) -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return


def test_list_comprehension_list_of_constant():

    @cudaq.kernel
    def kernel1() -> bool:
        combined = [[True] for _ in range(5)]
        res = [False for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[-1]

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel1)

    @cudaq.kernel
    def kernel2() -> float:
        combined = [[1.0] for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[-1]

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)

    @cudaq.kernel
    def kernel3() -> float:
        combined = [[1.j] for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[-1].imag

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    # keep after assert, such that we have no output if assert fails
    print(kernel3)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1
# CHECK-SAME: () -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3..
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return


def test_list_comprehension_list_of_variable():

    @cudaq.kernel
    def kernel1() -> bool:
        c = True
        combined = [[c] for _ in range(5)]
        res = [False for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[2]

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel2() -> float:
        c = 1.0
        combined = [[c] for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[2]

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel3() -> float:
        c = 1j
        combined = [[c] for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[2].imag

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1
# CHECK-SAME: () -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return


def test_list_comprehension_list_of_capture():

    c = True

    @cudaq.kernel
    def kernel1() -> bool:
        combined = [[c] for _ in range(5)]
        res = [False for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[3]

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel1)

    c = 1.0

    @cudaq.kernel
    def kernel2() -> float:
        combined = [[c] for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[3]

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)

    c = 1j

    @cudaq.kernel
    def kernel3() -> float:
        combined = [[c] for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[3].imag

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    # keep after assert, such that we have no output if assert fails
    print(kernel3)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1..
# CHECK-SAME: (%[[VAL_0:.*]]: i1 {quake.pylifted}) -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: (%[[VAL_0:.*]]: f64 {quake.pylifted}) -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3
# CHECK-SAME: (%[[VAL_45:.*]]: complex<f64> {quake.pylifted}) -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return


def test_list_comprehension_variable_list():

    @cudaq.kernel
    def kernel1() -> bool:
        c = [True]
        combined = [c for _ in range(5)]
        res = [False for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[1]

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel2() -> float:
        c = [1.0]
        combined = [c for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[1]

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel3() -> float:
        c = [1j]
        combined = [c for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[1].imag

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1
# CHECK-SAME: () -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return


def test_list_comprehension_capture_list():

    c = [True]

    @cudaq.kernel
    def kernel1() -> bool:
        combined = [c for _ in range(5)]
        res = [False for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[1]

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel1)

    c = [1.0]

    @cudaq.kernel
    def kernel2() -> float:
        combined = [c for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[1]

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)

    c = [1j]

    @cudaq.kernel
    def kernel3() -> float:
        combined = [c for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
            idx += 1
        return res[1].imag

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == 1.)
    # keep after assert, such that we have no output if assert fails
    print(kernel3)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1
# CHECK-SAME: (%[[VAL_0:.*]]: !cc.stdvec<i1> {quake.pylifted}) -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: (%[[VAL_0:.*]]: !cc.stdvec<f64> {quake.pylifted}) -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3
# CHECK-SAME: (%[[VAL_45:.*]]: !cc.stdvec<complex<f64>> {quake.pylifted}) -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return


def test_list_comprehension_tuple():
    print("test_list_comprehension_tuple:")

    @dataclass(slots=True)
    class MyTuple:
        first: float
        second: float

    @cudaq.kernel
    def kernel1() -> bool:
        first_idx = [i for i in range(3)]
        snd_idx = [i for i in range(3, 6)]
        pairs = [(first_idx[i], snd_idx[i]) for i in range(3)]
        correct = True
        for v1, v2 in pairs:
            correct = correct and (v2 - v1 == 3)
        return correct

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel1)

    @cudaq.kernel
    def kernel2():
        first_idx = [i for i in range(3)]
        snd_idx = [i for i in range(3, 6)]
        pairs = [(first_idx[i], snd_idx[i]) for i in range(3)]
        qs = cudaq.qvector(6)
        x(qs[0:len(first_idx)])
        [x.ctrl(qs[i1], qs[i2]) for i1, i2 in pairs]

    out = cudaq.sample(kernel2)
    assert (len(out) == 1 and '111111' in out)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)

    @cudaq.kernel
    def get_MyTuple() -> MyTuple:
        return MyTuple(0., 1.)

    @cudaq.kernel
    def kernel3() -> MyTuple:
        combined = [get_MyTuple() for _ in range(5)]
        res = MyTuple(0., 0.)
        for v in combined:
            res = MyTuple(res.first + v.first, res.second + v.second)
        return res

    # For reasons that are entirely unclear to me, using cudaq.run here
    # leads to an error in RecordLogParser.cpp: "Tuple size mismatch in
    # kernel and label." This only occurs when run with pytest, and the same
    # exact case in a separate tests below works just fine. No idea what's
    # going on.
    out = kernel3()
    assert (out == MyTuple(0., 5.))
    # keep after assert, such that we have no output if assert fails
    print(kernel3)

    @cudaq.kernel
    def kernel4() -> bool:
        # [(i, i + 3) for i in range(3)] is currently not supported
        # since inferring the type of an expression i + 3 is not implemented
        pairs = [(i, i + 3) for i in range(3)]
        correct = True
        for v1, v2 in pairs:
            correct = correct and (v2 - v1 == 3)
        return correct

    out = cudaq.run(kernel4, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel4)


# CHECK-LABEL: test_list_comprehension_tuple:
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1
# CHECK-SAME: () -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL: func.func @__nvqpp__mlirgen__kernel1..
# CHECK-SAME: .run() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, quake.cudaq_run = [i1]} {
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3
# CHECK-SAME: ) -> !cc.struct<"MyTuple" {f64, f64}> attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel4
# CHECK-SAME: () -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return


def test_list_comprehension_indirect_tuple():
    print("test_list_comprehension_indirect_tuple:")

    @dataclass(slots=True)
    class MyTuple:
        first: float
        second: float

    @cudaq.kernel
    def get_MyTuple() -> MyTuple:
        return MyTuple(0., 1.)

    @cudaq.kernel
    def kernel3d() -> MyTuple:
        combined = [get_MyTuple() for _ in range(5)]
        res = MyTuple(0., 0.)
        for v in combined:
            res = MyTuple(res.first + v.first, res.second + v.second)
        return res

    out = cudaq.run(kernel3d, shots_count=1)
    assert (len(out) == 1 and out[0] == MyTuple(0., 5.))
    # keep after assert, such that we have no output if assert fails
    print(kernel3d)


# CHECK-LABEL: test_list_comprehension_indirect_tuple:
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3d
# CHECK-SAME: ) -> !cc.struct<"MyTuple" {f64, f64}> attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return


def test_list_comprehension_call():
    print("test_list_comprehension_call:")

    @dataclass(slots=True)
    class MyTuple:
        first: float
        second: float

    @cudaq.kernel
    def kernel1() -> float:
        combined = [complex(0, 1) for _ in range(5)]
        res = 0.j
        for v in combined:
            res += v
        return res.imag

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == 5.)
    # keep after assert, such that we have no output if assert fails
    print(kernel1)

    @cudaq.kernel
    def kernel2() -> int:
        c = 1.0
        combined = [int(c) for _ in range(5)]
        res = 0
        for v in combined:
            res += v
        return res

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 5)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)

    @cudaq.kernel
    def kernel3() -> MyTuple:
        combined = [MyTuple(0., 1.) for _ in range(5)]
        res = MyTuple(0., 0.)
        for v in combined:
            res = MyTuple(res.first + v.first, res.second + v.second)
        return res

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == MyTuple(0., 5.))
    # keep after assert, such that we have no output if assert fails
    print(kernel3)

    @cudaq.kernel
    def get_float() -> float:
        return 1.

    @cudaq.kernel
    def kernel4() -> float:
        combined = [get_float() for _ in range(5)]
        res = 0.
        for v in combined:
            res += v
        return res

    out = cudaq.run(kernel4, shots_count=1)
    assert (len(out) == 1 and out[0] == 5.)
    # keep after assert, such that we have no output if assert fails
    print(kernel4)

    @cudaq.kernel
    def kernel5() -> bool:
        q = cudaq.qvector(6)
        x(q)
        res = [mz(r) for r in q]
        return res[3]

    out = cudaq.run(kernel5, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel5)

    @cudaq.kernel
    def kernel6() -> int:
        q = cudaq.qvector(6)
        x(q[0:3])
        res = [mz(r) for r in q]
        ires = 0
        for idx, v in enumerate(res):
            ires = ires | (int(v) << idx)
        return ires

    out = cudaq.run(kernel6, shots_count=1)
    assert (len(out) == 1 and out[0] == 7)
    # keep after assert, such that we have no output if assert fails
    print(kernel6)


# CHECK-LABEL: test_list_comprehension_call:
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: () -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3
# CHECK-SAME: () -> !cc.struct<"MyTuple" {f64, f64}> attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel4
# CHECK-SAME: ) -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel5
# CHECK-SAME: () -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel6
# CHECK-SAME: () -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"
# CHECK: return


def test_list_comprehension_void():
    print("test_list_comprehension_void:")

    @cudaq.kernel
    def kernel1():
        q = cudaq.qvector(6)
        [h(r) for r in q]
        x(q[0])
        x.ctrl(q[1], q[2])

    print(kernel1)

    @cudaq.kernel
    def apply_x(q: cudaq.qubit) -> None:
        return x(q)

    @cudaq.kernel
    def kernel2():
        q1 = cudaq.qvector(3)
        q2 = cudaq.qvector(3)
        [apply_x(r) for r in q1]
        for i in range(3):
            x.ctrl(q1[i], q2[i])

    out = cudaq.sample(kernel2)
    print("out is", out)
    assert (len(out) == 1 and '111111' in out)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)


# CHECK-LABEL: test_list_comprehension_void:
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: ) attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return


def test_list_comprehension_expressions():
    print("test_list_comprehension_expressions:")

    @cudaq.kernel
    def kernel1():
        orig = [True, False, True, False]
        # Sanity check to make sure we correctly process the type of i
        negated = [i for i in orig]
        qs = cudaq.qvector(8)
        for idx, v in enumerate(orig):
            if v:
                x(qs[idx])
        for idx, v in enumerate(negated):
            if v:
                x(qs[idx + len(orig)])

    out = cudaq.sample(kernel1)
    assert (len(out) == 1 and '10101010' in out)
    # keep after assert, such that we have no output if assert fails
    print(kernel1)

    @cudaq.kernel
    def kernel2():
        orig = [True, False, True, False]
        negated = [not i for i in orig]
        qs = cudaq.qvector(8)
        for idx, v in enumerate(orig):
            if v:
                x(qs[idx])
        for idx, v in enumerate(negated):
            if v:
                x(qs[idx + len(orig)])

    out = cudaq.sample(kernel2)
    assert (len(out) == 1 and '10100101' in out)
    # keep after assert, such that we have no output if assert fails
    print(kernel2)

    @cudaq.kernel
    def kernel3():
        orig = [True, False, True, False]
        # Sanity check to make sure we correctly process the type of i
        negated = [i for i in orig[1:3]]
        qs = cudaq.qvector(8)
        for idx, v in enumerate(orig):
            if v:
                x(qs[idx])
        for idx, v in enumerate(negated):
            if v:
                x(qs[idx + len(orig)])

    out = cudaq.sample(kernel3)
    assert (len(out) == 1 and '10100100' in out)
    # keep after assert, such that we have no output if assert fails
    print(kernel3)

    @cudaq.kernel
    def kernel4() -> int:
        q = cudaq.qvector(6)
        res = [mz(r) for r in q[0:3]]
        return len(res)

    out = cudaq.run(kernel4, shots_count=1)
    assert (len(out) == 1 and out[0] == 3)
    # keep after assert, such that we have no output if assert fails
    print(kernel4)

    @cudaq.kernel
    def kernel5() -> bool:
        vals = [0.5, 0.8, 0.1]
        correct = [v < 1.0 for v in vals]
        for c in correct:
            if not c:
                return False
        return True

    out = cudaq.run(kernel5, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    # keep after assert, such that we have no output if assert fails
    print(kernel5)

    @cudaq.kernel
    def kernel6() -> float:
        vals1 = [0.5, 1.0, 1.5]
        ang = [v * vals1[idx] for idx, v in enumerate(vals1[1:])]
        sum = 0.
        for a in ang:
            sum += a
        return sum

    out = cudaq.run(kernel6, shots_count=1)
    assert (len(out) == 1 and out[0] == 2.)
    # keep after assert, such that we have no output if assert fails
    print(kernel6)

    @cudaq.kernel
    def apply_rotations(qs: cudaq.qvector, angle: float, adj: list[bool]):
        for idx, is_adj in enumerate(adj):
            if is_adj:
                ry(-angle, qs[idx])
            else:
                ry(angle, qs[idx])

    @cudaq.kernel
    def kernel7(adj: list[bool]):
        targets = cudaq.qvector(len(adj))
        apply_rotations(targets, 1, [not b for b in adj])

    out = cudaq.sample(kernel7, [True, False, True])
    assert len(out) == 8
    # keep after assert, such that we have no output if assert fails
    print(kernel7)


# CHECK-LABEL: test_list_comprehension_expressions:
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel4
# CHECK-SAME: () -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel5
# CHECK-SAME: () -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel6
# CHECK-SAME: () -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK: return
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel7
# CHECK-SAME: (%[[VAL_0:.*]]: !cc.stdvec<i1>, %[[VAL_1:.*]]: !cc.callable<(!quake.veq<?>, f64, !cc.stdvec<i1>) -> ()> {quake.pylifted}) attributes {"cudaq-entrypoint", "cudaq-kernel"
# CHECK: return


def test_list_comprehension_failures():
    print("test_list_comprehension_failures:")
    try:

        @cudaq.kernel
        def kernel1() -> float:
            combined = [1.0 for _ in range(5)]
            res = 0
            for v in combined:
                res += v
            return res

        print(kernel1)
    except Exception as e:
        print("Exception kernel1:")
        print(e)

    try:

        @cudaq.kernel
        def kernel2() -> int:
            q = cudaq.qvector(6)
            x(q)
            # not supported, otherwise we would need to check for things like
            # mz(q[i:])
            res = [mz(q[i]) for i in range(3)]
            return len(res)

        print(kernel2)
    except Exception as e:
        print("Exception kernel2:")
        print(e)

    try:

        @cudaq.kernel
        def kernel3() -> bool:
            q = cudaq.qvector(6)
            x(q)
            res = [mz([r]) for r in q]
            return len(res)

        print(kernel3)
    except Exception as e:
        print("Exception kernel3:")
        print(e)

    try:

        @cudaq.kernel
        def kernel4() -> bool:
            q = cudaq.qvector(6)
            x(q)
            res = [mz(q) for r in q]
            return len(res)

        print(kernel4)
    except Exception as e:
        print("Exception kernel4:")
        print(e)

    try:

        @cudaq.kernel
        def kernel5() -> bool:
            vals = [[] for _ in range(3)]
            if vals[0] == [(1, 2)]:
                return False
            return True

        print(kernel5)
    except Exception as e:
        print("Exception kernel5:")
        print(e)

    @dataclass(slots=True)
    class MyTuple:
        first: float
        second: float

    try:

        @cudaq.kernel
        def kernel6() -> MyTuple:
            cvals = [1j for _ in range(3)]
            vals = [MyTuple(0, v) for v in cvals]
            res = MyTuple(0, 0)
            for v1, v2 in vals:
                res = MyTuple(res.first + v1, res.second + v2)
            return res

        print(kernel6)
    except Exception as e:
        print("Exception kernel6:")
        print(e)

    try:

        @cudaq.kernel
        def kernel7() -> int:
            v = (5, 1)
            l = [0. for _ in range(v)]
            return len(l)

        print(kernel7)
    except Exception as e:
        print("Exception kernel7:")
        print(e)

    try:

        @cudaq.kernel
        def kernel8() -> int:
            l = [0. for _ in range(1.)]
            return len(l)

        print(kernel8)
    except Exception as e:
        print("Exception kernel8:")
        print(e)


# CHECK-LABEL: test_list_comprehension_failures:
# CHECK-LABEL:  Exception kernel1:
# CHECK:        augment-assign must not change the variable type
# CHECK-NEXT:   (offending source -> res += v)

# CHECK-LABEL:  Exception kernel2:
# CHECK:        measurements in list comprehension expressions {{.*}} only supported when iterating over a vector of qubits
# CHECK-NEXT:   (offending source -> [mz(q[i]) for i in range(3)])

# CHECK-LABEL:  Exception kernel3:
# CHECK:        unsupported argument to measurement in list comprehension
# CHECK-NEXT:   (offending source -> [mz([r]) for r in q])

# CHECK-LABEL:  Exception kernel4:
# CHECK:        unsupported argument to measurement in list comprehension
# CHECK-NEXT:   (offending source -> [mz(q) for r in q])

# CHECK-LABEL:  Exception kernel5:
# CHECK:        creating empty lists is not supported in CUDA-Q
# CHECK-NEXT:   (offending source -> [{{.*}} for _ in range(3)])

# CHECK-LABEL:  Exception kernel6:
# CHECK:        cannot convert value of type complex<f64> to the requested type f64
# CHECK-NEXT:   (offending source -> MyTuple(0, v))

# CHECK-LABEL:  Exception kernel7:
# CHECK:        non-integer value in range expression
# CHECK-NEXT:   (offending source -> v)

# CHECK-LABEL:  Exception kernel8:
# CHECK:        non-integer value in range expression
# CHECK-NEXT:   (offending source -> 1.0)
