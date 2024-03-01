import cudaq
from cudaq import spin

# [Begin OptionA]
# A kernel that performs an X-gate on a provided qubit.
x_kernel, input_qubit = cudaq.make_kernel(cudaq.qubit)
x_kernel.x(input_qubit)

# A kernel that will call `x_kernel` as a controlled operation.
kernel = cudaq.make_kernel()
control_vector = kernel.qalloc(2)
target = kernel.qalloc()
kernel.x(control_vector)
kernel.x(target)
kernel.x(control_vector[1])
kernel.control(x_kernel, control_vector, target)

results = cudaq.sample(kernel)
print(results)
# [End OptionA]

# [Begin OptionB]
kernel = cudaq.make_kernel()
qvector = kernel.qalloc(3)

kernel.x(qvector)
kernel.x(qvector[1])
kernel.cx(control=[qvector[0], qvector[1]], target=qvector[2])

results = cudaq.sample(kernel)
print(results)
# [End OptionB]
