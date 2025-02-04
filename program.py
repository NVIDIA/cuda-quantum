# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

@cudaq.kernel
def opt(qubits: cudaq.qview, p_occ: int, q_occ: int):
    i_occ = 0
    j_occ = 0
    if (p_occ < q_occ):
        i_occ = p_occ
        j_occ = q_occ

    elif (p_occ > q_occ):
        i_occ = q_occ
        j_occ = p_occ

    for i in range(i_occ, j_occ):
        x.ctrl(qubits[i], qubits[i + 1])

@cudaq.kernel
def test():
    qubits = cudaq.qvector(6)
    x(qubits[0])

    n = 2
    arr0 = [i * 2 for i in range(n)]
    lenOccA = len(arr0)

    counter = 0
    nEle = 0
    for p in range(lenOccA - 1):
        for q in range(p + 1, lenOccA):
            nEle = nEle + 1

    counter = 0
    arr10 = [0 for k in range(nEle)]
    arr11 = [0 for k in range(nEle)]
    for p in range(lenOccA - 1):
        for q in range(p + 1, lenOccA):
            arr10[counter] = arr0[p]
            arr11[counter] = arr0[q]
            counter = counter + 1

    for i in range(len(arr10)):
        opt(qubits, arr10[i], arr11[i])

#cudaq.set_target('ionq', emulate="true")
counts = cudaq.sample(test)
assert len(counts) == 1
assert "111000" in counts
print(counts)