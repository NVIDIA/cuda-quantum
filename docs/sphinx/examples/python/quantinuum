import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`
cudaq.set_target("quantinuum")

# Create the kernel we'd like to execute on Quantinuum
kernel = cudaq.make_kernel()
qubits = kernel.qalloc(2)
kernel.h(qubits[0])
kernel.cx(qubits[0], qubits[1])
kernel.mz(qubits[0])
kernel.mz(qubits[1])
# Print out the kernel instructions before submitting.
print(kernel)

# Option A: Execute on Quantinuum and print out the results.
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from Quantinuum.
counts = cudaq.sample(kernel)
print(counts)

# Option B: Execute on Quantinuum and print out the results.
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by Quantinuum. This is ideal when submitting via a queue over
# the cloud.
future = cudaq.sample_async(kernel)
# ... more classical code to run ...
async_counts = future.get()

print(async_counts)
