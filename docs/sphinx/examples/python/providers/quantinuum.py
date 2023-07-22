import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# By default, we will submit to the Quantinuum syntax checker.
cudaq.set_target("quantinuum")

# Create the kernel we'd like to execute on Quantinuum
kernel = cudaq.make_kernel()
qubits = kernel.qalloc(2)
kernel.h(qubits[0])
kernel.cx(qubits[0], qubits[1])
kernel.mz(qubits[0])
kernel.mz(qubits[1])

# Submit to Quantinuum's endpoint and confirm the program is valid.

# Option A:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from Quantinuum.
# For now, we'll just use the synchronous call to get a syntax
# check from Quantinuum.
syntax_check = cudaq.sample(kernel)
if (syntax_check):
    print("Syntax check passed! Kernel is ready for submission.")

# Now we can update the target to the Quantinuum emulator and
# execute our program.
cudaq.set_target("quantinuum", machine="H1-2E")

# Option B:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by Quantinuum. This is ideal when submitting via a queue over
# the cloud.
future = cudaq.sample_async(kernel)
# ... more classical code to run ...
async_counts = future.get()
print(async_counts)

# We can also convert the future to a string and write it to file.
file = open("future.txt", "w")
file.write(str(future))
file.close()

# This allows us to grab the file at a later time and convert it
# back to a `cudaq::AsyncSampleResult`
same_file = open("future.txt", "r")
same_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

same_async_counts = same_async_results.get()
print(same_async_counts)
