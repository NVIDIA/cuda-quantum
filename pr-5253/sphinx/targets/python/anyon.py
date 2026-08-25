import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
# To use different targets in the same file, you must update
# it via another call to `cudaq.set_target()`

# To use the Anyon target you will need to set up credentials in `~/.anyon_config`
# The configuration file should contain your Anyon Technologies username and password:
# credentials: {"username":"<username>","password":"<password>"}

# Set the target to the default QPU
cudaq.set_target("anyon")

# You can specify a specific machine via the machine parameter:
# ```
# cudaq.set_target("anyon", machine="telegraph-8q")
# ```
# or for the larger system:
# ```
# cudaq.set_target("anyon", machine="berkeley-25q")
# ```


# Create the kernel we'd like to execute on Anyon.
@cudaq.kernel
def ghz():
    """Maximally entangled state between 5 qubits."""
    q = cudaq.qvector(5)
    h(q[0])
    for i in range(4):
        x.ctrl(q[i], q[i + 1])
    return mz(q)


# Execute on Anyon and print out the results.

# Option A (recommended):
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# remotely on Anyon's superconducting QPU. This is ideal for
# longer running jobs.
future = cudaq.sample_async(ghz)
# ... classical optimization code can run while job executes ...

# Can write the future to file:
with open("future.txt", "w") as outfile:
    print(future, file=outfile)

# Then come back and read it in later.
with open("future.txt", "r") as infile:
    restored_future = cudaq.AsyncSampleResult(infile.read())

# Get the results of the restored future.
async_counts = restored_future.get()
print("Asynchronous results:")
async_counts.dump()

# Option B:
# By using the synchronous `cudaq.sample`, the kernel
# will be executed on Anyon and the calling thread will be blocked
# until the results are returned.
counts = cudaq.sample(ghz)
print("\nSynchronous results:")
counts.dump()
