# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import pytest

cudaq.set_target("nvidia-dynamics")


class JaynesCummingsModel:
    wc = 1.0 * 2 * np.pi  # cavity frequency
    wa = 1.0 * 2 * np.pi  # atom frequency
    g = 0.25 * 2 * np.pi  # coupling strength

    kappa = 0.015  # cavity dissipation rate
    gamma = 0.15  # atom dissipation rate

    def __init__(self, size):
        self.fock_size = size // 2
        self.dimensions = {0: self.fock_size, 1: 2}
        self.a = operators.annihilate(0)
        self.a_dag = operators.create(0)
        self.number_op = operators.number(0)
        self.sm = pauli.minus(1)
        self.sp = pauli.plus(1)
        self.sz = pauli.z(1)

    def initial_state(self):
        cavity_state = cp.zeros(self.fock_size, dtype=cp.complex128)
        qubit_state = cp.array([1. / np.sqrt(2), 1. / np.sqrt(2)],
                               dtype=cp.complex128)
        return cudaq.State.from_data(cp.kron(cavity_state, qubit_state))

    def hamiltonian(self):
        return self.wc * self.number_op + (self.wa / 2.0) * self.sz + self.g * (
            self.a_dag * self.sm + self.a * self.sp)

    def collapse_operators(self):
        return [np.sqrt(self.kappa) * self.a, np.sqrt(self.gamma) * self.sm]

    def observables(self):
        return [self.number_op]


class CavityModel:
    kappa = 1.0
    eta = 1.5
    wc = 1.8
    wl = 2.0
    delta_c = wl - wc
    alpha0 = 0.3 - 0.5j

    def __init__(self, size):
        self.fock_size = size
        self.dimensions = {0: self.fock_size}
        self.a = operators.annihilate(0)
        self.a_dag = operators.create(0)
        self.number_op = operators.number(0)

    def initial_state(self):
        return cudaq.State.from_data(coherent_state(self.fock_size,
                                                    self.alpha0))

    def hamiltonian(self):
        return self.delta_c * self.number_op + self.eta * (self.a + self.a_dag)

    def collapse_operators(self):
        return [np.sqrt(self.kappa) * self.a]

    def observables(self):
        return [self.number_op]


class QubitChainModel:
    Jx = 0.2 * np.pi
    Jy = 0.2 * np.pi
    Jz = 0.2 * np.pi
    # dephasing rate
    gamma = 0.02

    def __init__(self, size):
        self.num_qubits = int(np.log2(size))
        self.dimensions = {}
        for i in range(self.num_qubits):
            self.dimensions[i] = 2

    def initial_state(self):
        q0_state = cp.array([0.0, 1.0], dtype=cp.complex128)
        other_qubit_state = cp.zeros(2**(self.num_qubits - 1),
                                     dtype=cp.complex128)
        other_qubit_state[0] = 1.0
        return cudaq.State.from_data(cp.kron(q0_state, other_qubit_state))

    def hamiltonian(self):
        H = operators.zero()

        for i in range(self.num_qubits):
            H -= 0.5 * 2 * np.pi * pauli.z(i)

        # Interaction terms
        for n in range(self.num_qubits - 1):
            H += -0.5 * self.Jx * pauli.x(n) * pauli.x(n + 1)
            H += -0.5 * self.Jy * pauli.y(n) * pauli.y(n + 1)
            H += -0.5 * self.Jz * pauli.z(n) * pauli.z(n + 1)

        return H

    def collapse_operators(self):
        return [
            np.sqrt(self.gamma) * pauli.z(i) for i in range(self.num_qubits)
        ]

    def observables(self):
        return []


@pytest.fixture(params=np.logspace(3, 7, 5, base=2, dtype=int).tolist())
def size(request):
    return request.param


@pytest.fixture(params=["Cavity", "Jaynes-Cummings", "Qubit Spin Chain"])
def model_solve(request):
    return request.param


def test_evolve(benchmark, model_solve, size):
    benchmark.group = "solvers:master-equation"
    steps = np.linspace(0, 20, 80)
    schedule = Schedule(steps, ["time"])
    if model_solve == "Cavity":
        model = CavityModel(size)
    elif model_solve == "Jaynes-Cummings":
        model = JaynesCummingsModel(size)
    elif model_solve == "Qubit Spin Chain":
        model = QubitChainModel(size)

    benchmark(evolve,
              model.hamiltonian(),
              model.dimensions,
              schedule,
              model.initial_state(),
              observables=model.observables(),
              collapse_operators=model.collapse_operators(),
              store_intermediate_results=False,
              integrator=ScipyZvodeIntegrator())


if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
