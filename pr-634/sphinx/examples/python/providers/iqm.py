import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`
cudaq.set_target("iqm",
                 url="http://localhost/cocos",
                 **{"qpu-architecture": "Adonis"})

# Adonis QPU architecture:
#       QB1
#        |
# QB2 - QB3 - QB4
#        |
#       QB5

# Create the kernel we'd like to execute on IQM.
kernel = cudaq.make_kernel()
qubits = kernel.qalloc(5)
kernel.h(qubits[2])  # QB3
kernel.cx(qubits[2], qubits[0])
kernel.mz(qubits)

# Execute on IQM Server and print out the results.

# Option A:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by IQM Server. This is ideal when submitting via a queue over
# the cloud.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

# We can either retrieve the results later in the program with
# ```
# async_counts = async_results.get()
# ```
# or we can also write the job reference (`async_results`) to
# a file and load it later or from a different process.
file = open("future.txt", "w")
file.write(str(async_results))
file.close()

# We can later read the file content and retrieve the job
# information and results.
same_file = open("future.txt", "r")
retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

counts = retrieved_async_results.get()
print(counts)

# Option B:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from IQM Server.
counts = cudaq.sample(kernel)
print(counts)
