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

    @cudaq.kernel
    def kernel1() -> float:
        l = [1,2,3]
        l[0] = 4
        c = complex(0,0)
        c += 1
        res = l.size + c.real
        for v in l:
            res += v
        return res 

    out = cudaq.run(kernel1, shots_count=1)
    assert(len(out) == 1 and out[0] == 13)
    print("[attribute access] kernel 1 outputs " + str(out[0]))

# CHECK-LABEL: [attribute access] kernel 1 outputs 13.0

test_attribute_access()

def test_attribute_failures():

    @cudaq.kernel
    def kernel1() -> int:
        l = [1,2,3]
        l[0] = 4
        l.size = 4
        return len(l)

    try:
        print(kernel1)
    except Exception as e:
        print("Exception kernel1:")
        print(e)

# CHECK-LABEL:  Exception kernel1:
# CHECK:        invalid CUDA-Q attribute assignment
# CHECK-NEXT:   (offending source -> l.size = 4)
