import cudaq

# Here we demonstrate how one might apply multi-controlled
# operations on a general CUDA-Q kernel.


# [Begin OptionA]
# A kernel that performs an X-gate on a provided qubit.
@cudaq.kernel
def x_kernel(qubit: cudaq.qubit):
    x(qubit)


# A kernel that will call `x_kernel` as a controlled operation.
@cudaq.kernel
def kernel():
    control_vector = cudaq.qvector(2)
    target = cudaq.qubit()
    x(control_vector)
    x(target)
    x(control_vector[1])
    cudaq.control(x_kernel, control_vector, target)


results = cudaq.sample(kernel)
print(results)
# [End OptionA]


# [Begin OptionB]
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(3)
    x(qvector)
    x(qvector[1])
    x.ctrl([qvector[0], qvector[1]], qvector[2])
    mz(qvector)


results = cudaq.sample(kernel)
print(results)
# [End OptionB]
