import cudaq

# This example assumes the NVQC API key has been set in the `NVQC_API_KEY` environment variable.
# If not, you can set the API Key environment variable in the Python script with:
# ```
# os.environ["NVQC_API_KEY"] = "<YOUR NVQC API KEY>"`
# ```

cudaq.set_target("nvqc")

num_qubits = 20
kernel = cudaq.make_kernel()
qubits = kernel.qalloc(num_qubits)
# Place qubits in GHZ state.
kernel.h(qubits[0])
for i in range(num_qubits - 1):
    kernel.cx(qubits[i], qubits[i + 1])

state = cudaq.get_state(kernel)
print("Amplitude(00..00) =", state[0])
print("Amplitude(11..11) =", state[2**num_qubits - 1])
