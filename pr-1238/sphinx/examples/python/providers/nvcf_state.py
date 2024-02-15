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
#                 api_key="<YOUR NVCF API KEY>"
#                 function_id="<YOUR NVCF FUNCTION ID>")
# ```
cudaq.set_target("nvcf")

num_qubits = 20


@cudaq.kernel(jit=True)
def ghz(num_qubits: int):
    qubits = cudaq.qvector(num_qubits)
    # Place qubits in GHZ state.
    h(qubits[0])
    for i in range(num_qubits - 1):
        x.ctrl(qubits[i], qubits[i + 1])


state = cudaq.get_state(ghz, num_qubits)
print("Amplitude(00..00) =", state[0])
print("Amplitude(11..11) =", state[2**num_qubits - 1])
