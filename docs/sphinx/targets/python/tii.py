import cudaq
import os

# Set the target at the beginning of the program.
cudaq.set_target("tii",
                 device="tii-sim",
                 project=os.environ.get("TII_PROJECT", None))


# Create the kernel.
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])
    mz(qvector)


# Note: Increase shots count for better distribution of results.
# Using small number here for testing purposes.
SHOTS = 50

# Execute on synchronously on the TII cloud and print out the results.
counts = cudaq.sample(kernel, shots_count=SHOTS)
print(counts)
