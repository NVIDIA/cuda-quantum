# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os
import pytest
import cudaq


# Check that the `qubitMeasurementFeedback` metadata is properly set.
def test_conditional_on_vars():

    @cudaq.kernel
    def test1():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        bit = mz(ancilla[0])
        if bit:
            x(data[0])
        mz(data)

    print(test1)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test1() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test2():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        bit = mz(ancilla[0])
        if bit == False:
            x(data[0])
        mz(data)

    print(test2)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test2() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test3():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        bits = mz(ancilla)
        if bits[0]:
            x(data[0])
        mz(data)

    print(test3)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test4():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        bits = mz(ancilla)
        if bits[0] != True:
            x(data[0])
        mz(data)

    print(test4)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test4() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test5():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        bits = mz(ancilla)
        if not bits[0]:
            x(data[0])
        mz(data)

    print(test5)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test5() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test6():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        bits = mz(ancilla)
        if bits[0] and bits[1]:
            x(data[0])
        mz(data)

    print(test6)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test6() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {


def test_conditional_on_measure():

    @cudaq.kernel
    def test7():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        if mz(ancilla[0]):
            x(data[0])
        mz(data)

    print(test7)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test7() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test8():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        if mz(ancilla[0]) == False:
            x(data[0])
        mz(data)

    print(test8)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test8() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test9():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        if not mz(ancilla[0]):
            x(data[0])
        mz(data)

    print(test9)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test9() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test10():
        data = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        if mz(ancilla[0]) or mz(ancilla[1]):
            x(data[0])
        mz(data)

    print(test10)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test10() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test11():
        data = cudaq.qubit()
        res = mz(data)
        flag = res

        if flag:
            h(data)

    print(test11)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test11() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test12():
        data = cudaq.qubit()
        res = mz(data)
        flag1 = res
        flag2 = flag1

        if flag2 == False:
            x(data)

    print(test12)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test12() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test13():
        qubits = cudaq.qvector(2)
        h(qubits)
        m0 = mz(qubits[0])
        m1 = mz(qubits[1])

        if m0 or m1:
            x.ctrl(qubits[0], qubits[1])
            res = mz(qubits[0])

    print(test13)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test13() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {

    @cudaq.kernel
    def test14():
        qubits = cudaq.qvector(2)
        h(qubits)
        m0 = mz(qubits[0])
        m1 = mz(qubits[1])

        if m0 == True:
            if m1:
                x.ctrl(qubits[0], qubits[1])
                res = mz(qubits[0])

    print(test14)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__test14() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
