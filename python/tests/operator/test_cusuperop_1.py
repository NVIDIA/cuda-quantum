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

hamiltonian = 2 * np.pi * 0.1 * pauli.x(0)
num_qubits = 1
dimensions = {0: 2}
rho0_ = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128)
rho0_ = cp.asfortranarray(rho0_)
# TODO: need to create a new CUDAQ state implementation for cuSuperOp
rho0 = DenseDensityMatrix(ctx, rho0_)

steps = numpy.linspace(0, 10, 101)
schedule = Schedule(steps, ["time"])

evolution_result = evolve(hamiltonian, dimensions, schedule, rho0, observables = [pauli.y(0), pauli.z(0)], collapse_operators = [], store_intermediate_results = True)
# close the work stream
ctx.free()


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9, 6))
plt.plot(steps, evolution_result.expect[0])
plt.plot(steps, evolution_result.expect[1])
plt.ylabel('Expectation value')
plt.xlabel('Time')
plt.legend(("Sigma-Y", "Sigma-Z"))
fig.savefig('example_1.png', dpi=fig.dpi)