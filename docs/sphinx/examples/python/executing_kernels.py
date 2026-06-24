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
''' [Begin `SampleOutput`]  
     ╭───╮     
q0 : ┤ h ├──●──
     ╰───╯╭─┴─╮
q1 : ─────┤ x ├
          ╰───╯

{ 11:506 00:494 }
 [End `SampleOutput`] '''

# [Begin `SampleAsync`]
result_async = cudaq.sample_async(kernel, qubit_count, shots_count=1000)

print(result_async.get())
# [End `SampleAsync`]
''' [Begin `SampleAsyncOutput`]
{ 00:498 11:502 }
[End `SampleAsyncOutput`] '''


# [Begin Run]
# Define a quantum kernel that returns an integer
@cudaq.kernel
def simple_ghz(num_qubits: int) -> int:
    # Allocate qubits
    qubits = cudaq.qvector(num_qubits)

    # Create GHZ state
    h(qubits[0])
    for i in range(1, num_qubits):
        x.ctrl(qubits[0], qubits[i])

    # Measure and return total number of qubits in state |1⟩
    res = 0
    for i in range(num_qubits):
        if mz(qubits[i]):
            res += 1

    return res


# Execute the kernel 20 times
num_qubits = 3
results = cudaq.run(simple_ghz, num_qubits, shots_count=20)

print(f"Executed {len(results)} shots")
print(f"Results: {results}")
print(f"Possible values: Either 0 or {num_qubits} due to GHZ state properties")

# Count occurrences of each result
value_counts = {}
for value in results:
    value_counts[value] = value_counts.get(value, 0) + 1

print("\nCounts of each result:")
for value, count in value_counts.items():
    print(f"{value}: {count} times")
# [End Run]
''' [Begin `RunOutput`]  
Executed 20 shots
Results: [0, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 0, 3, 0, 3, 3, 0, 3, 3, 3]
Possible values: Either 0 or 3 due to GHZ state properties

Counts of each result:
0: 8 times
3: 12 times
 [End `RunOutput`] '''

# [Begin `RunCustom`]
from dataclasses import dataclass


# Define a custom `dataclass` to return from our quantum kernel
@dataclass(slots=True)
class MeasurementResult:
    first_qubit: bool
    last_qubit: bool
    total_ones: int


@cudaq.kernel
def bell_pair_with_data() -> MeasurementResult:
    # Create a bell pair
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])

    # Measure both qubits
    first_result = mz(qubits[0])
    last_result = mz(qubits[1])

    # Return custom data structure with results
    total = 0
    if first_result:
        total = 1
    if last_result:
        total = total + 1

    return MeasurementResult(first_result, last_result, total)


# Run the kernel 10 times and get all results
results = cudaq.run(bell_pair_with_data, shots_count=10)

# Analyze the results
print("Individual measurement results:")
for i, res in enumerate(results):
    print(
        f"Shot {i}: {{{res.first_qubit}, {res.last_qubit}}}\ttotal ones={res.total_ones}"
    )

# Verify the Bell state correlations
correlated_count = sum(
    1 for res in results if res.first_qubit == res.last_qubit)
print(
    f"\nCorrelated measurements: {correlated_count}/{len(results)} ({correlated_count/len(results)*100:.1f}%)"
)
# [End `RunCustom`]
''' [Begin `RunCustomOutput`]
Individual measurement results:
Shot 0: {True, True}	total ones=2
Shot 1: {False, False}	total ones=0
Shot 2: {False, False}	total ones=0
Shot 3: {False, False}	total ones=0
Shot 4: {True, True}	total ones=2
Shot 5: {True, True}	total ones=2
Shot 6: {True, True}	total ones=2
Shot 7: {False, False}	total ones=0
Shot 8: {False, False}	total ones=0
Shot 9: {True, True}	total ones=2

Correlated measurements: 10/10 (100.0%)
 [End `RunCustomOutput`] '''


# [Begin `RunAsync`]
# Example of `run_async` with a simple integer return type
# Define a quantum kernel that returns an integer
@cudaq.kernel
def simple_count(angle: float) -> int:
    q = cudaq.qubit()
    rx(angle, q)
    return int(mz(q))


# Execute asynchronously with different parameters
futures = []
angles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

for i, angle in enumerate(angles):
    futures.append(cudaq.run_async(simple_count, angle, shots_count=10))

# Process results as they complete
for i, future in enumerate(futures):
    results = future.get()
    ones_count = sum(results)
    print(f"Angle {angles[i]:.1f}: {ones_count}/10 ones measured")
# [End `RunAsync`]
''' [Begin `RunAsyncOutput`]
Angle 0.0: 0/10 ones measured
Angle 0.2: 0/10 ones measured
Angle 0.4: 0/10 ones measured
Angle 0.6: 0/10 ones measured
Angle 0.8: 1/10 ones measured
Angle 1.0: 2/10 ones measured
Angle 1.2: 3/10 ones measured
Angle 1.4: 5/10 ones measured
 [End `RunAsyncOutput`] '''

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
''' [Begin `ObserveOutput`]  
<H> = 0.0
 [End `ObserveOutput`] '''

# [Begin `GetState`]
# Compute the statevector of the kernel
result = cudaq.get_state(kernel, qubit_count)

print(np.array(result.dump()))
# [End `GetState`]
''' [Begin `GetStateOutput`]
[0.+0.j 0.+0.j 0.+0.j 1.+0.j]
 [End `GetStateOutput`] '''

# [Begin `GetStateAsync`]
import numpy as np


@cudaq.kernel
def bell_state():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


# Get state asynchronously
state_future = cudaq.get_state_async(bell_state)

# Do other work while waiting for state computation...
print("Computing state asynchronously...")

# Get the state when ready
state = state_future.get()
print("Bell state vector:")
print(np.array(state.dump()))
# [End `GetStateAsync`]
''' [Begin `GetStateAsyncOutput`]
Computing state asynchronously...
Bell state vector:
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
 [End `GetStateAsyncOutput`] '''


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
