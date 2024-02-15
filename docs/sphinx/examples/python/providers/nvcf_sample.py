import cudaq

# This example assumes the NVCF API key and Function Id have been set in the `~/.nvcf_config` file/environment variables.
# If not, you can set the API Key and Function ID environment variables in the Python script with:
# ```
# os.environ["NVCF_API_KEY"] = "<YOUR NVCF API KEY>"`
# os.environ["NVCF_FUNCTION_ID"] = "<YOUR NVCF FUNCTION ID>"
# ```
# Alternatively, the `api_key` and `function_id` values can be passed to the target directly,
# ```
# cudaq.set_target("nvcf",
#                 backend="tensornet",
#                 api_key="<YOUR NVCF API KEY>"
#                 function_id="<YOUR NVCF FUNCTION ID>")
# ```
cudaq.set_target("nvcf", backend="tensornet")

num_qubits = 50


@cudaq.kernel(jit=True)
def ghz(num_qubits: int):
    qubits = cudaq.qvector(num_qubits)
    # Place qubits in GHZ state.
    h(qubits[0])
    for i in range(num_qubits - 1):
        x.ctrl(qubits[i], qubits[i + 1])
    # Measure.
    mz(qubits)


counts = cudaq.sample(ghz, num_qubits, shots_count=100)
print(counts)
