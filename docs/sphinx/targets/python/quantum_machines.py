import traceback
import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`
cudaq.set_target("quantum_machines")


# Create the kernel we'd like to execute
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])
    mz(qvector)


try:
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
except Exception as ex:
    print(ex)
    print(traceback.format_exc())