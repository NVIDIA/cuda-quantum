# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

from dataclasses import dataclass
import cudaq


def test_attribute_access():

    @dataclass(slots=True)
    class MyTuple:
        control: cudaq.qubit
        targets: cudaq.qview

    # TODO: this is a good example to reexamine some of the
    # handling in the bridge; the Python AST does represent LoadOp
    # and StoreOp, which we are not currently overloading.
    # Creating explicit overloads for these could allow some clean up.

    @cudaq.kernel
    def kernel1() -> float:
        l = [1, 2, 3]
        l[0] = 4
        c = complex(0, 0)
        c += 1
        res = l.size + c.real
        for v in l:
            res += v
        return res

    out = cudaq.run(kernel1, shots_count=1)
    assert (len(out) == 1 and out[0] == 13)
    print("[attribute access] kernel 1 outputs " + str(out[0]))

    @cudaq.kernel
    def kernel2():
        qs = MyTuple(cudaq.qubit(), cudaq.qvector(3))
        x(qs.targets.front())

    out = cudaq.sample(kernel2, shots_count=100)
    assert (len(out) == 1 and out.most_probable() == '0100')
    print("[attribute access] kernel 2 outputs " + out.most_probable())

    @cudaq.kernel
    def kernel3():
        qs = MyTuple(cudaq.qubit(), cudaq.qvector(3))
        x(qs.targets.back())

    out = cudaq.sample(kernel3, shots_count=100)
    assert (len(out) == 1 and out.most_probable() == '0001')
    print("[attribute access] kernel 3 outputs " + out.most_probable())


# CHECK-LABEL: [attribute access] kernel 1 outputs 13.0
# CHECK-LABEL: [attribute access] kernel 2 outputs 0100
# CHECK-LABEL: [attribute access] kernel 3 outputs 0001


def test_attribute_failures():

    @cudaq.kernel
    def kernel1() -> int:
        l = [1, 2, 3]
        l[0] = 4
        l.size = 4
        return len(l)

    try:
        print(kernel1)
    except Exception as e:
        print("Failure kernel1:")
        print(e)

    @cudaq.kernel
    def kernel2():
        qs = cudaq.qvector(2)
        qs.append(cudaq.qubit())
        x(qs)

    try:
        print(kernel2)
    except Exception as e:
        print("Failure kernel2:")
        print(e)

    @cudaq.kernel
    def kernel3():
        angles = [0.5, 1.]
        angles.append(1.5)
        q = cudaq.qubit()
        for a in angles:
            rz(a, q)

    try:
        print(kernel3)
    except Exception as e:
        print("Failure kernel3:")
        print(e)


# CHECK-LABEL:  Failure kernel1:
# CHECK:        attribute expression does not produce a modifiable value
# CHECK-NEXT:   (offending source -> l.size)

# CHECK-LABEL:  Failure kernel2:
# CHECK:        CUDA-Q does not allow dynamic resizing or lists, arrays, or qvectors.
# CHECK-NEXT:   (offending source -> qs.append(cudaq.qubit()))

# CHECK-LABEL:  Failure kernel3:
# CHECK:        CUDA-Q does not allow dynamic resizing or lists, arrays, or qvectors.
# CHECK-NEXT:   (offending source -> angles.append(1.5))
