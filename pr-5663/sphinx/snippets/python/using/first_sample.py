# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq


# Define our kernel.
@cudaq.kernel
def kernel(qubit_count: int):
    # Allocate our qubits.
    qvector = cudaq.qvector(qubit_count)
    # Place the first qubit in the superposition state.
    h(qvector[0])
    # Loop through the allocated qubits and apply controlled-X,
    # or CNOT, operations between them.
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit + 1])
    # Measure the qubits.
    mz(qvector)
    # [End Documentation]


#[Begin Sample1]
qubit_count = 2
print(cudaq.draw(kernel, qubit_count))
results = cudaq.sample(kernel, qubit_count)
# Should see a roughly 50/50 distribution between the |00> and
# |11> states. Example: {00: 505  11: 495}
print("Measurement distribution:" + str(results))
#[End Sample1]

#[Begin Sample2]
# With an increased shots count, we will still see the same 50/50 distribution,
# but now with 10,000 total measurements instead of the default 1000.
# Example: {00: 5005  11: 4995}
results = cudaq.sample(kernel, qubit_count, shots_count=10000)
print("Measurement distribution:" + str(results))
#[End Sample2]

#[Begin Sample3]
most_probable_result = results.most_probable()
probability = results.probability(most_probable_result)
print("Most probable result: " + most_probable_result)
print("Measured with probability " + str(probability), end='\n\n')
#[End Sample3]

# Parallelize over the various kernels one would like to execute.


#[Begin SampleAsync]
@cudaq.kernel
def kernel2(qubit_count: int):
    # Allocate our qubits.
    qvector = cudaq.qvector(qubit_count)
    # Place all qubits in a uniform superposition.
    h(qvector)
    # Measure the qubits.
    mz(qvector)


num_gpus = cudaq.num_available_gpus()
if num_gpus > 1:
    # Set the target to include multiple virtual QPUs.
    cudaq.set_target("nvidia", option="mqpu")
    # Asynchronous execution on multiple virtual QPUs, each simulated by an NVIDIA GPU.
    result_1 = cudaq.sample_async(kernel,
                                  qubit_count,
                                  shots_count=1000,
                                  qpu_id=0)
    result_2 = cudaq.sample_async(kernel2,
                                  qubit_count,
                                  shots_count=1000,
                                  qpu_id=1)
else:
    # Schedule for execution on the same virtual QPU.
    result_1 = cudaq.sample_async(kernel,
                                  qubit_count,
                                  shots_count=1000,
                                  qpu_id=0)
    result_2 = cudaq.sample_async(kernel2,
                                  qubit_count,
                                  shots_count=1000,
                                  qpu_id=0)

print("Measurement distribution for kernel:" + str(result_1.get()))
print("Measurement distribution for kernel2:" + str(result_2.get()))
#[End SampleAsync]
