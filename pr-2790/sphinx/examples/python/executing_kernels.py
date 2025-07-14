# [Begin Sample]
import cudaq
import numpy as np

qubit_count = 2

# Define the simulation target.
cudaq.set_target("qpp-cpu")


# Define a quantum kernel function.
@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)

    # 2-qubit GHZ state.
    h(qvector[0])
    for i in range(1, qubit_count):
        x.ctrl(qvector[0], qvector[i])

    # If we do not specify measurements, all qubits are measured in
    # the Z-basis by default or we can manually specify it also
    mz(qvector)


print(cudaq.draw(kernel, qubit_count))

result = cudaq.sample(kernel, qubit_count, shots_count=1000)

print(result)
# [End Sample]
''' [Begin SampleOutput]  
     ╭───╮     
q0 : ┤ h ├──●──
     ╰───╯╭─┴─╮
q1 : ─────┤ x ├
          ╰───╯

{ 11:506 00:494 }
 [End SampleOutput] '''

# [Begin `SampleAsync`]
result_async = cudaq.sample_async(kernel, qubit_count, shots_count=1000)

print(result_async.get())
# [End `SampleAsync`]
''' [Begin `SampleAsyncOutput`]
{ 00:498 11:502 }
[End `SampleAsyncOutput`] '''

# [Begin Observe]
from cudaq import spin

# Define a Hamiltonian in terms of Pauli Spin operators.
hamiltonian = spin.z(0) + spin.y(1) + spin.x(0) * spin.z(0)


@cudaq.kernel
def kernel1(n_qubits: int):
    qubits = cudaq.qvector(n_qubits)
    h(qubits[0])
    for i in range(1, n_qubits):
        x.ctrl(qubits[0], qubits[i])


# Compute the expectation value given the state prepared by the kernel.
result = cudaq.observe(kernel1, hamiltonian, qubit_count).expectation()

print('<H> =', result)
# [End Observe]
''' [Begin ObserveOutput]  
<H> = 0.0
 [End ObserveOutput] '''

# [Begin `GetState`]
# Compute the statevector of the kernel
result = cudaq.get_state(kernel, qubit_count)

print(np.array(result))
# [End `GetState`]
''' [Begin `GetStateOutput`]
[0.+0.j 0.+0.j 0.+0.j 1.+0.j]
 [End `GetStateOutput`] '''


# [Begin `ObserveAsync`]
# Define a quantum kernel function.
@cudaq.kernel
def kernel1(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)

    # 2-qubit GHZ state.
    h(qvector[0])
    for i in range(1, qubit_count):
        x.ctrl(qvector[0], qvector[i])


# Measuring the expectation value of 2 different Hamiltonians in parallel
hamiltonian_1 = spin.x(0) + spin.y(1) + spin.z(0) * spin.y(1)

# Asynchronous execution on multiple `qpus` via `nvidia` `gpus`.
result_1 = cudaq.observe_async(kernel1, hamiltonian_1, qubit_count, qpu_id=0)

# Retrieve results
print(result_1.get().expectation())
# [End `ObserveAsync`]
''' [Begin `ObserveAsyncOutput`]  
1.1102230246251565e-16
 [End `ObserveAsyncOutput`] '''
