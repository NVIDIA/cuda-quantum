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
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel2() -> float:
        combined = [1.0 for _ in range(5)]
        res = 0.
        for v in combined:
            res += v
        return res

    out = cudaq.run(kernel2, shots_count=1)
    assert (len(out) == 1 and out[0] == 5.0)
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel3() -> float:
        combined = [1.j for _ in range(5)]
        res = 0.j
        for v in combined:
            res += v
        return res.imag

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == 5.0)
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant 1 : i8
# CHECK:            %[[VAL_1:.*]] = cc.alloca !cc.array<i8 x 5>
# CHECK:            %[[VAL_3:.*]] = cc.compute_ptr %[[VAL_1]][{{.*}}] : (!cc.ptr<!cc.array<i8 x 5>>, i64) -> !cc.ptr<i8>
# CHECK:            cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i8>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f64
# CHECK:            %[[VAL_1:.*]] = cc.alloca !cc.array<f64 x 5>
# CHECK:            %[[VAL_2:.*]] = cc.compute_ptr %[[VAL_1]][{{.*}}] : (!cc.ptr<!cc.array<f64 x 5>>, i64) -> !cc.ptr<f64>
# CHECK:            cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<f64>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = complex.constant [0.000000e+00, 1.000000e+00] : complex<f64>
# CHECK:            %[[VAL_1:.*]] = cc.alloca !cc.array<complex<f64> x 5>
# CHECK:            %[[VAL_2:.*]] = cc.compute_ptr %[[VAL_1]][{{.*}}] : (!cc.ptr<!cc.array<complex<f64> x 5>>, i64) -> !cc.ptr<complex<f64>>
# CHECK:            cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<complex<f64>>


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
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant true
# CHECK:            %[[VAL_1:.*]] = cc.alloca i1
# CHECK:            cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i1>
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<i8 x 5>
# CHECK:            %[[VAL_4:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i1>
# CHECK:            %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<i8 x 5>>, i64) -> !cc.ptr<i8>
# CHECK:            %[[VAL_6:.*]] = cc.cast unsigned %[[VAL_4]] : (i1) -> i8
# CHECK:            cc.store %[[VAL_6]], %[[VAL_5]] : !cc.ptr<i8>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f64
# CHECK:            %[[VAL_1:.*]] = cc.alloca f64
# CHECK:            cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<f64>
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<f64 x 5>
# CHECK:            %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<f64>
# CHECK:            %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<f64 x 5>>, i64) -> !cc.ptr<f64>
# CHECK:            cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<f64>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = complex.constant [0.000000e+00, 1.000000e+00] : complex<f64>
# CHECK:            %[[VAL_1:.*]] = cc.alloca complex<f64>
# CHECK:            cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<complex<f64>>
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<complex<f64> x 5>
# CHECK:            %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<complex<f64>>
# CHECK:            %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<complex<f64> x 5>>, i64) -> !cc.ptr<complex<f64>>
# CHECK:            cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<complex<f64>>


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
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant 1 : i8
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<i8 x 5>
# CHECK:            %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<i8 x 5>>, i64) -> !cc.ptr<i8>
# CHECK:            cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i8>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f64
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<f64 x 5>
# CHECK:            %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<f64 x 5>>, i64) -> !cc.ptr<f64>
# CHECK:            cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<f64>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = complex.constant [0.000000e+00, 1.000000e+00] : complex<f64>
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<complex<f64> x 5>
# CHECK:            %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<complex<f64> x 5>>, i64) -> !cc.ptr<complex<f64>>
# CHECK:            cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<complex<f64>>


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
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant 1 : i8
# CHECK:            %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<!cc.stdvec<i1> x 5>
# CHECK:            %[[VAL_3:.*]] = cc.alloca !cc.array<i8 x 1>
# CHECK:            %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<i8 x 1>>) -> !cc.ptr<!cc.array<i8 x ?>>
# CHECK:            %[[VAL_5:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<i8 x 1>>) -> !cc.ptr<i8>
# CHECK:            cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<i8>
# CHECK:            %[[VAL_6:.*]] = cc.stdvec_init %[[VAL_4]], %[[VAL_1]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.stdvec<i1>
# CHECK:            %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<!cc.stdvec<i1> x 5>>, i64) -> !cc.ptr<!cc.stdvec<i1>>
# CHECK:            cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<!cc.stdvec<i1>>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-DAG:        %[[VAL_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:        %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f64
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<!cc.stdvec<f64> x 5>
# CHECK:            %[[VAL_3:.*]] = cc.alloca !cc.array<f64 x 1>
# CHECK:            %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 1>>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:            %[[VAL_5:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 1>>) -> !cc.ptr<f64>
# CHECK:            cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<f64>
# CHECK:            %[[VAL_6:.*]] = cc.stdvec_init %[[VAL_4]], %[[VAL_0]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.stdvec<f64>
# CHECK:            %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<!cc.stdvec<f64> x 5>>, i64) -> !cc.ptr<!cc.stdvec<f64>>
# CHECK:            cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<!cc.stdvec<f64>>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-DAG:        %[[VAL_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:        %[[VAL_1:.*]] = complex.constant [0.000000e+00, 1.000000e+00] : complex<f64>
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<!cc.stdvec<complex<f64>> x 5>
# CHECK:            %[[VAL_3:.*]] = cc.alloca !cc.array<complex<f64> x 1>
# CHECK:            %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<complex<f64> x 1>>) -> !cc.ptr<!cc.array<complex<f64> x ?>>
# CHECK:            %[[VAL_5:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<complex<f64> x 1>>) -> !cc.ptr<complex<f64>>
# CHECK:            cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<complex<f64>>
# CHECK:            %[[VAL_6:.*]] = cc.stdvec_init %[[VAL_4]], %[[VAL_0]] : (!cc.ptr<!cc.array<complex<f64> x ?>>, i64) -> !cc.stdvec<complex<f64>>
# CHECK:            %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<!cc.stdvec<complex<f64>> x 5>>, i64) -> !cc.ptr<!cc.stdvec<complex<f64>>>
# CHECK:            cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<!cc.stdvec<complex<f64>>>


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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}


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
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}


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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}


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
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}


def test_list_comprehension_tuple():

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
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

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
    # leads to an error in in RecordLogParser.cpp:
    # "Tuple size mismatch in kernel and label."
    # This only occurs when run with pytest, and the same exact case in
    # a separate tests below works just fine. No idea what's going on.
    out = kernel3()
    assert (out == MyTuple(0., 5.))
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel4
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> !cc.struct<"MyTuple" {f64, f64}> attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel4() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}


def test_list_comprehension_indirect_tuple():

    @dataclass(slots=True)
    class MyTuple:
        first: float
        second: float

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

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == MyTuple(0., 5.))
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> !cc.struct<"MyTuple" {f64, f64}> attributes {"cudaq-entrypoint", "cudaq-kernel"}


def test_list_comprehension_call():

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
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel3() -> MyTuple:
        combined = [MyTuple(0., 1.) for _ in range(5)]
        res = MyTuple(0., 0.)
        for v in combined:
            res = MyTuple(res.first + v.first, res.second + v.second)
        return res

    out = cudaq.run(kernel3, shots_count=1)
    assert (len(out) == 1 and out[0] == MyTuple(0., 5.))
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel4
         )  # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel5() -> bool:
        q = cudaq.qvector(6)
        x(q)
        res = [mz(r) for r in q]
        return res[3]

    out = cudaq.run(kernel5, shots_count=1)
    assert (len(out) == 1 and out[0] == True)
    print(kernel5
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel6
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() -> !cc.struct<"MyTuple" {f64, f64}> attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel4() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel5() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel6() -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"}


def test_list_comprehension_void():

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
    assert (len(out) == 1 and '111111' in out)
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 6 : i64
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<6>
# CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_7:.*]]: i64):
# CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_7]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_8]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_7]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           }
# CHECK:           %[[VAL_11:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<6>) -> !quake.ref
# CHECK:           quake.x %[[VAL_11]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_12:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<6>) -> !quake.ref
# CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_3]][2] : (!quake.veq<6>) -> !quake.ref
# CHECK:           quake.x {{\[}}%[[VAL_12]]] %[[VAL_13]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() attributes {"cudaq-entrypoint", "cudaq-kernel"}


def test_list_comprehension_expressions():

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
    print(kernel1
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel2
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel3
         )  # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel4() -> int:
        q = cudaq.qvector(6)
        res = [mz(r) for r in q[0:3]]
        return len(res)

    out = cudaq.run(kernel4, shots_count=1)
    assert (len(out) == 1 and out[0] == 3)
    print(kernel4
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel5
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel6
         )  # keep after assert, such that we have no output if assert fails

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
    print(kernel7
         )  # keep after assert, such that we have no output if assert fails


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel3() attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel4() -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel5() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel6() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel7(%arg0: !cc.stdvec<i1>) attributes {"cudaq-entrypoint", "cudaq-kernel"}


def test_list_comprehension_failures():

    @cudaq.kernel
    def kernel1() -> float:
        combined = [1.0 for _ in range(5)]
        res = 0
        for v in combined:
            res += v
        return res

    try:
        print(kernel1)
    except Exception as e:
        print("Exception kernel1:")
        print(e)

    @cudaq.kernel
    def kernel2() -> int:
        q = cudaq.qvector(6)
        x(q)
        # not supported, otherwise we would need to check for things like mz(q[i:])
        res = [mz(q[i]) for i in range(3)]
        return len(res)

    try:
        print(kernel2)
    except Exception as e:
        print("Exception kernel2:")
        print(e)

    @cudaq.kernel
    def kernel3() -> bool:
        q = cudaq.qvector(6)
        x(q)
        res = [mz([r]) for r in q]
        return len(res)

    try:
        print(kernel3)
    except Exception as e:
        print("Exception kernel3:")
        print(e)

    @cudaq.kernel
    def kernel4() -> bool:
        q = cudaq.qvector(6)
        x(q)
        res = [mz(q) for r in q]
        return len(res)

    try:
        print(kernel4)
    except Exception as e:
        print("Exception kernel4:")
        print(e)

    @cudaq.kernel
    def kernel5() -> bool:
        vals = [[] for _ in range(3)]
        if vals[0] == [(1, 2)]:
            return False
        return True

    try:
        print(kernel5)
    except Exception as e:
        print("Exception kernel5:")
        print(e)

    @dataclass(slots=True)
    class MyTuple:
        first: float
        second: float

    @cudaq.kernel
    def kernel6() -> MyTuple:
        cvals = [1j for _ in range(3)]
        vals = [MyTuple(0, v) for v in cvals]
        res = MyTuple(0, 0)
        for v1, v2 in vals:
            res = MyTuple(res.first + v1, res.second + v2)
        return res

    try:
        print(kernel6)
    except Exception as e:
        print("Exception kernel6:")
        print(e)

    @cudaq.kernel
    def kernel7() -> int:
        v = (5, 1)
        l = [0. for _ in range(v)]
        return len(l)

    try:
        print(kernel7)
    except Exception as e:
        print("Exception kernel7:")
        print(e)

    @cudaq.kernel
    def kernel8() -> int:
        l = [0. for _ in range(1.)]
        return len(l)

    try:
        print(kernel8)
    except Exception as e:
        print("Exception kernel8:")
        print(e)


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
