import cudaq, inspect, numpy, operator, types, uuid
from cudaq.operator import *
from typing import Any, Optional

import numpy as np
import cupy as cp
from cusuperop import (
    optimize_strides,
    tensor_product,        # function for constructing tensor products of elementary tensor operators
    CallbackTensor,
    DenseDensityMatrix,    # mixed quantum state represented by the full density matrix
    GeneralOperator,
    Operator,              # quantum many-body operator (super-operator)
    OperatorTerm,
    OperatorAction,        # right-hand side of the desired master equation
    WorkStream             # work stream
)

ctx = WorkStream()

# Time-dependent Hamiltonian
# Square-pulse
def square_pulse(t, args = None):
    # print("Callback @ t =", t)
    if (t >=2) & (t <= 4):
        return 1.0
    else:
        return 0.0
    
squared_pulse = ScalarOperator(square_pulse)
hamiltonian = squared_pulse * pauli.x(0)
dimensions = {0: 2}
rho0_ = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128)
rho0_ = cp.asfortranarray(rho0_)
# TODO: need to create a new CUDAQ state implementation for cuSuperOp
rho0 = DenseDensityMatrix(ctx, rho0_)

steps = numpy.linspace(0, 10, 101)
schedule = Schedule(steps, ["time"])

evolution_result = evolve(hamiltonian, dimensions, schedule, rho0, observables = [operators.number(0)], collapse_operators = [], store_intermediate_results = True)
# close the work stream
ctx.free()

pulse_values = [square_pulse(t) for t in steps]

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9, 6))
plt.plot(steps, evolution_result.expect[0])
plt.plot(steps, pulse_values)
plt.ylabel('Expectation value')
plt.xlabel('Time')
plt.legend(("Population in |1>", "Drive Pulse"))
fig.savefig('example_3.png', dpi=fig.dpi)