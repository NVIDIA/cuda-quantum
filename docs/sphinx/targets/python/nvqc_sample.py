import cudaq

# This example assumes the NVQC API key has been set in the `NVQC_API_KEY` environment variable.
# If not, you can set the API Key environment variable in the Python script with:
# ```
# os.environ["NVQC_API_KEY"] = "<YOUR NVQC API KEY>"`
# ```
cudaq.set_target("nvqc", backend="tensornet")

# Note: The `tensornet` simulator is capable of distributing tensor contraction operations across multiple GPUs.
# User can use the `ngpus` option to target a multi-GPU NVQC endpoint.
# For example, to use the `tensornet` simulator with 8 GPUs, we can do
# `cudaq.set_target("nvqc", backend="tensornet", ngpus=8)`
# Please refer to your NVQC dashboard for the list of available multi-GPU configurations.
num_qubits = 50
kernel = cudaq.make_kernel()
qubits = kernel.qalloc(num_qubits)
# Place qubits in superposition state.
kernel.h(qubits[0])
for i in range(num_qubits - 1):
    kernel.cx(qubits[i], qubits[i + 1])
# Measure.
kernel.mz(qubits)

counts = cudaq.sample(kernel, shots_count=100)
print(counts)
