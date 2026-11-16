# [Begin Definition]
import cudaq


@cudaq.kernel
def kernel():
    A = cudaq.qubit()
    B = cudaq.qvector(3)
    C = cudaq.qvector(5)


# [End Definition]

# [Begin `InputDefinition`]
N = 2


@cudaq.kernel
def kernel(N: int):
    register = cudaq.qvector(N)


# [End `InputDefinition`]

# [Begin `PassingComplexVector`]
# Passing complex vectors as parameters
c = [.707 + 0j, 0 - .707j]


@cudaq.kernel
def kernel(vec: list[complex]):
    q = cudaq.qvector(vec)


# [End `PassingComplexVector`]

# [Begin `CapturingComplexVector`]
# Capturing complex vectors
c = [0.70710678 + 0j, 0., 0., 0.70710678]


@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)


# [End `CapturingComplexVector`]

# [Begin `PrecisionAgnosticAPI`]
# Precision-Agnostic API
import numpy as np

c = np.array([0.70710678 + 0j, 0., 0., 0.70710678], dtype=cudaq.complex())


@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)


# [End `PrecisionAgnosticAPI`]

# [Begin `CUDAQAmplitudes`]
# Define as CUDA-Q amplitudes
c = cudaq.amplitudes([0.70710678 + 0j, 0., 0., 0.70710678])


@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)


# [End `CUDAQAmplitudes`]

# [Begin `PassingState`]
# Pass in a state from another kernel
c = [0.70710678 + 0j, 0., 0., 0.70710678]


@cudaq.kernel
def kernel_initial():
    q = cudaq.qvector(c)


state_to_pass = cudaq.get_state(kernel_initial)


@cudaq.kernel
def kernel(state: cudaq.State):
    q = cudaq.qvector(state)


kernel(state_to_pass)
# [End `PassingState`]


# [Begin `AllQubits`]
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    h(register)


# [End `AllQubits`]


# [Begin `IndividualQubits`]
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    h(register[0])  # first qubit
    h(register[-1])  # last qubit


# [End `IndividualQubits`]


# [Begin `ControlledOperations`]
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    x.ctrl(register[0],
           register[1])  # CNOT gate applied with qubit 0 as control


# [End `ControlledOperations`]


# [Begin `MultiControlledOperations`]
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    x.ctrl([register[0], register[1]],
           register[2])  # X applied to qubit two controlled by qubit 0 and 1


# [End `MultiControlledOperations`]


# [Begin `ControlledKernel`]
@cudaq.kernel
def x_kernel(qubit: cudaq.qubit):
    x(qubit)


# A kernel that will call `x_kernel` as a controlled operation.
@cudaq.kernel
def kernel():

    control_vector = cudaq.qvector(2)
    target = cudaq.qubit()

    x(control_vector)
    x(target)
    x(control_vector[1])
    cudaq.control(x_kernel, control_vector, target)


# The above is equivalent to:
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(3)
    x(qvector)
    x(qvector[1])
    x.ctrl([qvector[0], qvector[1]], qvector[2])
    mz(qvector)


results = cudaq.sample(kernel)
print(results)
# [End `ControlledKernel`]


# [Begin `AdjointOperations`]
@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    t.adj(register[0])


# [End `AdjointOperations`]

# [Begin `CustomOperations`]
import numpy as np

cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))


@cudaq.kernel
def kernel():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    custom_x(qubits[0])
    custom_x.ctrl(qubits[0], qubits[1])


# [End `CustomOperations`]


# [Begin `BuildingKernelsWithKernels`]
@cudaq.kernel
def kernel_A(qubit_0: cudaq.qubit, qubit_1: cudaq.qubit):
    x.ctrl(qubit_0, qubit_1)


@cudaq.kernel
def kernel_B():
    reg = cudaq.qvector(10)
    for i in range(5):
        kernel_A(reg[i], reg[i + 1])


#[End `BuildingKernelsWithKernels`]


# [Begin `ParameterizedKernels`]
@cudaq.kernel
def kernel(thetas: list[float]):
    qubits = cudaq.qvector(2)
    rx(thetas[0], qubits[0])
    ry(thetas[1], qubits[1])


thetas = [.024, .543]
kernel(thetas)
# [End `ParameterizedKernels`]
