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


@cudaq.kernel
def ghz(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    # Place qubits in GHZ state.
    h(qvector[0])
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit + 1])
    # Measure.
    mz(qvector)


qubit_count = 50
counts = cudaq.sample(ghz, qubit_count, shots_count=100)
print(counts)
