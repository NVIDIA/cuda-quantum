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
    assert(len(out) == 1 and out[0] == True)
    print(kernel1) # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel2() -> float:
        combined = [1.0 for _ in range(5)]
        res = 0.
        for v in combined:
            res += v
        return res

    out = cudaq.run(kernel2, shots_count=1)
    assert(len(out) == 1 and out[0] == 5.0)
    print(kernel2) # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel3() -> float:
        combined = [1.j for _ in range(5)]
        res = 0.j
        for v in combined:
            res += v
        return res.imag
    
    out = cudaq.run(kernel3, shots_count=1)
    assert(len(out) == 1 and out[0] == 5.0)
    print(kernel3) # keep after assert, such that we have no output if assert fails

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant true
# CHECK:            %[[VAL_1:.*]] = cc.alloca !cc.array<i1 x 5> 
# CHECK:            %[[VAL_2:.*]] = cc.compute_ptr %[[VAL_1]][{{.*}}] : (!cc.ptr<!cc.array<i1 x 5>>, i64) -> !cc.ptr<i1>
# CHECK:            cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i1>

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
    assert(len(out) == 1 and out[0] == True)
    print(kernel1) # keep after assert, such that we have no output if assert fails

    @cudaq.kernel
    def kernel2() -> int:
        c = 1.0
        combined = [c for _ in range(5)]
        res = 0.
        for v in combined:
            res += v
        return int(res)

    out = cudaq.run(kernel2, shots_count=1)
    assert(len(out) == 1 and out[0] == 5)
    print(kernel2) # keep after assert, such that we have no output if assert fails
    

    @cudaq.kernel
    def kernel3() -> float:
        c = 1.j
        combined = [c for _ in range(5)]
        res = 0.j
        for v in combined:
            res += v
        return res.imag
    
    out = cudaq.run(kernel3, shots_count=1)
    assert(len(out) == 1 and out[0] == 5.0)
    print(kernel3) # keep after assert, such that we have no output if assert fails

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant true
# CHECK:            %[[VAL_1:.*]] = cc.alloca i1
# CHECK:            cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i1>
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<i1 x 5> 
# CHECK:            %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i1>
# CHECK:            %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<i1 x 5>>, i64) -> !cc.ptr<i1>
# CHECK:            cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i1>

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
    assert(len(out) == 1 and out[0] == True)
    print(kernel1) # keep after assert, such that we have no output if assert fails

    c = 1.0
    @cudaq.kernel
    def kernel2() -> float:
        combined = [c for _ in range(5)]
        res = 0.
        for v in combined:
            res += v
        return res
    
    out = cudaq.run(kernel2, shots_count=1)
    assert(len(out) == 1 and out[0] == 5.)
    print(kernel2) # keep after assert, such that we have no output if assert fails

    c = 1.j
    @cudaq.kernel
    def kernel3() -> float:
        combined = [c for _ in range(5)]
        res = 0.j
        for v in combined:
            res += v
        return res.imag
    
    out = cudaq.run(kernel3, shots_count=1)
    assert(len(out) == 1 and out[0] == 5.)
    print(kernel3) # keep after assert, such that we have no output if assert fails

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"}
# CHECK:            %[[VAL_0:.*]] = arith.constant true
# CHECK:            %[[VAL_1:.*]] = cc.alloca i1
# CHECK:            cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i1>
# CHECK:            %[[VAL_2:.*]] = cc.alloca !cc.array<i1 x 5> 
# CHECK:            %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i1>
# CHECK:            %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][{{.*}}] : (!cc.ptr<!cc.array<i1 x 5>>, i64) -> !cc.ptr<i1>
# CHECK:            cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i1>

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel2() -> f64 attributes {"cudaq-entrypoint", "cudaq-kernel"}
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


def test_failures():
    @cudaq.kernel
    def kernel() -> float:
        combined = [1.0 for _ in range(5)]
        res = 0 # FIXME: CREATE TEST THAT CHECKS WE GET AN ERROR IF THIS WAS INT
        for v in combined:
            res += v
        return res


'''
def test_list_comprehension():
    test_list_comprehension_constant()
    #test_list_comprehension_slice()
    return



def test_list_comprehension_gate_invocation():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(6)
        [h(r) for r in q]
        x(q[0])
        x.ctrl(q[1], q[2])

    print(kernel)

def test_list_comprehension_list_of_constant():

    @cudaq.kernel
    def kernel() -> bool:
        combined = [[False] for _ in range(5)]
        res = [True for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def kernel() -> float:
        combined = [[1.0] for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def kernel() -> float:
        combined = [[1.j] for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0].imag
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

def test_list_comprehension_list_of_variable():

    @cudaq.kernel
    def kernel() -> bool:
        c = False
        combined = [[c] for _ in range(5)]
        res = [True for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def kernel() -> float:
        c = 1.0
        combined = [[c] for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def kernel() -> float:
        c = 1j
        combined = [[c] for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0].imag
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

def test_list_comprehension_list_of_capture():

    c = False
    @cudaq.kernel
    def kernel() -> bool:
        combined = [[c] for _ in range(5)]
        res = [True for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    c = 1.0
    @cudaq.kernel
    def kernel() -> float:
        combined = [[c] for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    c = 1j
    @cudaq.kernel
    def kernel() -> float:
        combined = [[c] for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0].imag
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

def test_list_comprehension_variable_list():

    @cudaq.kernel
    def kernel() -> bool:
        c = [False]
        combined = [c for _ in range(5)]
        res = [True for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def kernel() -> float:
        c = [1.0]
        combined = [c for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def kernel() -> float:
        c = [1j]
        combined = [c for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0].imag
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

def test_list_comprehension_capture_list():

    c = [False]
    @cudaq.kernel
    def kernel() -> bool:
        combined = [c for _ in range(5)]
        res = [True for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    c = [1.0]
    @cudaq.kernel
    def kernel() -> float:
        combined = [c for _ in range(5)]
        res = [0. for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0]
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    c = [1j]
    @cudaq.kernel
    def kernel() -> float:
        combined = [c for _ in range(5)]
        res = [0j for _ in range(5)]
        idx = 0
        for v in combined:
            res[idx] = v[0]
        return res[0].imag
    
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

def test_list_comprehension_call():

    @dataclass(slots=True)
    class MyTuple:
        first: float
        second: float

    @cudaq.kernel
    def kernel() -> float:
        combined = [complex(0, 1) for _ in range(5)]
        res = 0.j
        for v in combined:
            res += v
        return res.imag

    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def kernel() -> int:
        c = 1.0
        combined = [int(c) for _ in range(5)]
        res = 0
        for v in combined:
            res += v
        return res

    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def kernel() -> MyTuple:
        combined = [MyTuple(0., 1.) for _ in range(5)]
        res = MyTuple(0., 0.)
        for v in combined:
            res = MyTuple(res.first + v.first, res.second + v.second)
        return res

    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def get_float() -> float:
        return 1.

    @cudaq.kernel
    def kernel() -> float:
        combined = [get_float() for _ in range(5)]
        res = 0.
        for v in combined:
            res += v
        return res

    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def apply_h(q : cudaq.qubit) -> None:
        return h(q)

    @cudaq.kernel
    def kernel():
        q1 = cudaq.qvector(3)
        q2 = cudaq.qvector(3)
        [apply_h(r) for r in q1]
        for i in range(3):
            x.ctrl(q1[i], q2[i])

    #print(kernel)
    out = cudaq.sample(kernel)
    print(out)

    """
    @cudaq.kernel
    def get_MyTuple() -> MyTuple:
        return MyTuple(0., 1.)

    @cudaq.kernel
    def kernel() -> MyTuple:
        combined = [get_MyTuple() for _ in range(5)] # FIXME: assert that this gives a nice error for now
        res = MyTuple(0., 0.)
        for v in combined:
            res = MyTuple(res.first + v.first, res.second + v.second)
        return res

    #print(kernel)
    out = cudaq.run(kernel)
    print(out)    

    @cudaq.kernel
    def kernel() -> bool:
        q = cudaq.qvector(6)
        h(q)
        res = [mz(q[i]) for i in range(3)] # FIXME: cannot be supported, otherwise we would need to check for mz(q[i:])
        return res[0]

    #print(kernel)
    out = cudaq.run(kernel)
    print(out)

    @cudaq.kernel
    def kernel() -> bool:
        q = cudaq.qvector(6)
        h(q)
        # also check that [mz(q) for _ in q] fails...
        #res = [mz([r]) for r in q] # FIXME: SHOULD FAIL
        #res = [mz(q) for _ in q]
        res = [mz(q[i:]) for i in range(3)]
        return res[0][0]

    #print(kernel)
    out = cudaq.run(kernel)
    print(out)
    """

    @cudaq.kernel
    def kernel() -> bool:
        q = cudaq.qvector(6)
        h(q)
        res = [mz(r) for r in q]
        return res[0]

    #print(kernel)
    out = cudaq.run(kernel)
    print(out)


def test_list_comprehension_slice():

    @cudaq.kernel
    def kernel() -> bool:
        orig = [True, False, True, False]
        # orig[0::2] -> step value in slice not supported
        # orig[0:2] -> valid iterable not detected
        combined = [i for i in orig] 
        res = False
        for v in combined:
            res = res or v
        return res
    #print(kernel)
    out = cudaq.run(kernel)
    print(out)


def test_list_comprehension_data_types():

    @cudaq.kernel
    def kernel() -> complex:
        q1 = cudaq.qvector(3)
        q2 = cudaq.qvector(2)
        h(q1)
        h(q2)
        res1 = mz(q1)
        res2 = mz(q2)
        c0 = complex(0, 0)
        combined = [False for _ in range(5)]
        #combined = [c0 for _ in range(5)]
        # combined = [complex(0, 0) for _ in range(5)] # FIXME: add test that this fails properly
        idx = 0
        for r in res1:
            combined[idx] = r
            idx += 1
        for r in res2:
            combined[idx] = r
        sum = complex(0, 0)
        for v in combined:
            sum = sum # + v
            # not supported
            # sum.real += v.real
            # sum.imag += v.imag
            # sum += v # FIXME: NOT SUPPORTED...
        return sum

    print(kernel)
    cudaq.run(kernel)

if __name__ == '__main__':
    test_list_comprehension()
'''