# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

cudaq.set_target("quantinuum", emulate=True)
def test_computations():
    @cudaq.kernel
    def kernel(n: int, m: np.int32):
        q = cudaq.qvector(n)
        j = 0
        jf = 1.2
        for i in range(10):
            k = 0 
            if i > 5:
                k = 1
            x(q[k])
            if mz(q[k]):
                j = j+ 1
                m = m + m
                jf = jf + jf
    
        if jf > 3 and j > 5:
            x(q[0])
    
    
    print(cudaq.sample(kernel, 2, 134))

test_computations()

def test_return():
    @cudaq.kernel
    def kernel(n: int) -> int:
        q = cudaq.qvector(n)
        if mz(q[0]):
            x(q[0])
            return 1
        return 0
    
    print(cudaq.sample(kernel, 2))

test_return()

def test_conditionally_terminating_loops():
    @cudaq.kernel
    def kernel(n: int):
        q = cudaq.qvector(n)
        for qubit in q:
            if mz(qubit):
                break
            else:
                x(qubit)

    print(cudaq.sample(kernel, 2))

test_conditionally_terminating_loops()

# define void @__nvqpp__mlirgen__kernel() local_unnamed_addr #0 {
# "0":
#   tail call void @__quantum__qis__mz__body(%Qubit* null, %Result* writeonly null)
#   tail call void @__quantum__rt__result_record_output(%Result* null, i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.72303030303000, i64 0, i64 0))
#   %0 = tail call i1 @__quantum__rt__read_result(%Result* null)
#   br i1 %0, label %"3", label %"1"

# "1":                                              ; preds = %"0"
#   tail call void @__quantum__qis__x__body(%Qubit* null)
#   tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull writeonly inttoptr (i64 1 to %Result*))
#   tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 1 to %Result*), i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.72303030303100, i64 0, i64 0))
#   %1 = tail call i1 @__quantum__rt__read_result(%Result* nonnull inttoptr (i64 1 to %Result*))
#   br i1 %1, label %"3", label %"2"

# "2":                                              ; preds = %"1"
#   tail call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
#   br label %"3"

# "3":                                              ; preds = %"2", %"1", %"0"
#   ret void
# }


# this example produces an unnecessary global array containing numbers from 0 to n-1:)
# def test_iterations():
#     @cudaq.kernel
#     def kernel(n: int):
#         q = cudaq.qvector(n)
#         i = 0
#         while i < n-1:
#             cx(q[i], q[i+1])
#             i = i + 1

#     print(cudaq.sample(kernel, 10000))

# test_iterations()

# define void @__nvqpp__mlirgen__kernel() local_unnamed_addr #0 {
# "0":
#   br label %"1"

# "1":                                              ; preds = %"0", %"1"
#   %0 = phi i64 [ 0, %"0" ], [ %5, %"1" ]
#   %1 = phi i64 [ 0, %"0" ], [ %3, %"1" ]
#   %2 = inttoptr i64 %0 to %Qubit*
#   %3 = add nuw nsw i64 %1, 1
#   %4 = getelementptr [10000 x i64], [10000 x i64]* @__nvqpp__mlirgen__kernel.rodata_0, i64 0, i64 %3
#   %5 = load i64, i64* %4, align 8
#   %6 = inttoptr i64 %5 to %Qubit*
#   tail call void @__quantum__qis__cnot__body(%Qubit* %2, %Qubit* %6)
#   %7 = icmp ult i64 %1, 9998
#   br i1 %7, label %"1", label %"2"

# "2":                                              ; preds = %"1"
#   ret void
# }