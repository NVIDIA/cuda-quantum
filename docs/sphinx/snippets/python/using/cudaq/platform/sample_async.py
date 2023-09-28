import cudaq

cudaq.set_target("nvidia-mqpu")
target = cudaq.get_target()
num_qpus = target.num_qpus()
print("Number of QPUs:", num_qpus)

kernel, runtime_param = cudaq.make_kernel(int)
qubits = kernel.qalloc(runtime_param)
# Place qubits in superposition state.
kernel.h(qubits)
# Measure.
kernel.mz(qubits)

count_futures = []
for qpu in range(num_qpus):
    count_futures.append(cudaq.sample_async(kernel, 5, qpu_id=qpu))

for counts in count_futures:
    print(counts.get())