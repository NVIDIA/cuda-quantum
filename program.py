# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

@cudaq.kernel
def bell_pair():
    # def k():
    #     q0 = cudaq.qvector(2)
    #     h(q0[0])
    #     cx(q0[0], q0[1])
    #     mz(q0)
    
    #k()
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])
    mz(q)

counts = cudaq.sample(bell_pair)
print(counts)

# print("*** EXISTING ***")

# print("*** DRAW")
# print(cudaq.draw(bell_pair))

# print("*** MLIR")
# print(bell_pair)

print("*** TO_QIR")
print(cudaq.to_qir(bell_pair))

print("*** TO_QIR with profile=qir-base")
print(cudaq.to_qir(bell_pair, profile="qir-base"))

print("*** TO_QIR with profile=qir-adaptive")
print(cudaq.to_qir(bell_pair, profile="qir-adaptive"))


print("*** NEW ***")

print("*** TRANSLATE to mlir")
print(cudaq.translate(bell_pair, format="mlir"))

print("*** TRANSLATE to qir")
print(cudaq.translate(bell_pair, format="qir"))

print("*** TRANSLATE to openqasm")
print(cudaq.translate(bell_pair, format="openqasm"))
