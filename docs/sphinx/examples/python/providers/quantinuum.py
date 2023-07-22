import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# By default, we will submit to the Quantinuum syntax checker.
cudaq.set_target("quantinuum")

# Create the kernel we'd like to execute on Quantinuum.
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
# after the job has been executed by the Quantinuum service.
# We will use the synchronous call to submit to the syntax
# checker to confirm the validity of the program.
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
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

# We can either retrieve the results later in the program with
# ```
# async_counts = async_results.get()
# ```
# or wee can also write the job reference (`async_results`) to
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
