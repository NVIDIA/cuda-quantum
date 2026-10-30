import cudaq
import os

# Set LQCLOUD_API_KEY in the environment before running this example. CUDA-Q
# reads the credential directly; do not put it in source or command-line flags.
# Select an available QPU by its exact name in https://cloud.logicalqubit.com.
# MQ02 is an example; export LQCLOUD_URL and LQCLOUD_QPU to override the defaults.
cudaq.set_target("lqcloud",
                 url=os.environ.get("LQCLOUD_URL",
                                    "https://cloud.logicalqubit.com"),
                 qpu=os.environ.get("LQCLOUD_QPU", "MQ02"))


@cudaq.kernel
def bell():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    mz(qubits)


counts = cudaq.sample(bell, shots_count=5)
print(counts)

# This future is consumed in the same process in which it was created.
async_counts = cudaq.sample_async(bell, shots_count=5).get()
print(async_counts)
