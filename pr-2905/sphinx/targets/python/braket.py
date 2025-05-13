import cudaq

# NOTE: Amazon Braket credentials must be set before running this program.
# Amazon Braket costs apply.
cudaq.set_target("braket")

# The default device is SV1, state vector simulator. Users may choose any of
# the available devices by supplying its `ARN` with the `machine` parameter.
# For example,
# ```
# cudaq.set_target("braket", machine="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
# ```


# Create the kernel we'd like to execute
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])
    mz(qvector)


# Execute and print out the results.

# Option A:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by Amazon Braket.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

async_counts = async_results.get()
print(async_counts)

# Option B:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from Amazon Braket.
counts = cudaq.sample(kernel)
print(counts)
