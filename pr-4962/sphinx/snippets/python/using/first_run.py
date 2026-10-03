# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin Run1]
import cudaq


# Define a quantum kernel that returns an integer
@cudaq.kernel
def ghz_kernel(qubit_count: int) -> int:
    # Allocate qubits
    qubits = cudaq.qvector(qubit_count)

    # Create GHZ state
    h(qubits[0])
    for i in range(1, qubit_count):
        x.ctrl(qubits[0], qubits[i])

    # Measure and count the number of qubits in state |1⟩
    result = 0
    for i in range(qubit_count):
        if mz(qubits[i]):
            result += 1

    return result


# Execute the kernel multiple times and collect individual results
qubit_count = 3
results = cudaq.run(ghz_kernel, qubit_count, shots_count=10)
print(f"Executed {len(results)} shots")
print(f"Results: {results}")
#[End Run1]

#[Begin Run2]
# Count occurrences of each result
value_counts = {}
for value in results:
    value_counts[value] = value_counts.get(value, 0) + 1

print("\nCounts of each result:")
for value, count in sorted(value_counts.items()):
    print(f"Result {value}: {count} times")

# Analyze patterns in the results
zero_count = results.count(0)
full_count = results.count(qubit_count)
other_count = len(results) - zero_count - full_count
print(f"\nGHZ state analysis:")
print(
    f"  All qubits in |0⟩: {zero_count} times ({zero_count/len(results)*100:.1f}%)"
)
print(
    f"  All qubits in |1⟩: {full_count} times ({full_count/len(results)*100:.1f}%)"
)
print(
    f"  Other states: {other_count} times ({other_count/len(results)*100:.1f}%)"
)
#[End Run2]


#[Begin RunAsync]
# Define a simple kernel for asynchronous execution
@cudaq.kernel
def simple_kernel(theta: float) -> bool:
    q = cudaq.qubit()
    rx(theta, q)
    return mz(q)


# Check if we have multiple GPUs
num_gpus = cudaq.num_available_gpus()
if num_gpus > 1:
    # Set the target to include multiple virtual QPUs
    cudaq.set_target("nvidia", option="mqpu")

    # Run kernels asynchronously with different parameters
    future1 = cudaq.run_async(simple_kernel, 0.0, shots_count=100, qpu_id=0)
    future2 = cudaq.run_async(simple_kernel, 3.14159, shots_count=100, qpu_id=1)
else:
    # Schedule for execution on the same virtual QPU, defaulting to `qpu_id=0`
    future1 = cudaq.run_async(simple_kernel, 0.0, shots_count=100)
    future2 = cudaq.run_async(simple_kernel, 3.14159, shots_count=100)

# Get results when ready
results1 = future1.get()
results2 = future2.get()

# Analyze the results
print("\nAsynchronous execution results:")
true_count1 = sum(1 for res in results1 if res)
true_count2 = sum(1 for res in results2 if res)
print(f"Kernel with theta=0.0: {true_count1}/100 times measured |1⟩")
print(f"Kernel with theta=π: {true_count2}/100 times measured |1⟩")
#[End RunAsync]
