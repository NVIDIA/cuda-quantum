# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import numpy as np
import cudaq


def num_hwe_parameters(numQubits, numLayers):
    """
    For the given number of qubits and layers, return the required number of `hwe` parameters.
    """
    return 2 * numQubits * (1 + numLayers)


def hwe(kernel, qubits, numQubits, numLayers, parameters, cnotCoupling=None):
    """
    Generate the hardware-efficient CUDA-Q kernel.

    Args:
        kernel (:class:`Kernel`): The existing `cudaq.Kernel` to append to
        qubits (:class:`qview`): Pre-allocated qubits
        `numQubits` (int): Number of qubits
        `numLayers` (int): Number of layers
        parameters (List[float]): The list of parameters
        `cnotCoupling` (Optional[List[tuple(int, int)]]): The `CNOT` couplers
        
    """
    if cnotCoupling == None:
        cnotCoupling = [(i, i + 1) for i in range(numQubits - 1)]

    thetaCounter = 0
    for i in range(numQubits):
        kernel.ry(parameters[thetaCounter], qubits[i])
        kernel.rz(parameters[thetaCounter + 1], qubits[i])
        thetaCounter = thetaCounter + 2

    for i in range(numLayers):
        for cnot in cnotCoupling:
            kernel.cx(qubits[cnot[0]], qubits[cnot[1]])
        for q in range(numQubits):
            kernel.ry(parameters[thetaCounter], qubits[q])
            kernel.rz(parameters[thetaCounter + 1], qubits[q])
            thetaCounter += 2
