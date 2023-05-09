import cudaq
from cudaq import spin

cudaq.list_qpus()
cudaq.set_qpu('cuquantum')

n_qubits = 1 
kernel, params = cudaq.make_kernel(list)
qubit = kernel.qalloc(n_qubits)

kernel.rx(params[0], qubit)
kernel.ry(params[1], qubit)

hamiltonian = spin.z(0)

def cost(params): 

    exp_vals = cudaq.observe(kernel, hamiltonian, params, shots_count = 1000).expectation_z()

    return exp_vals

optimizer = cudaq.optimizers.GradientDescent()

optimizer.initial_parameters = [1,2]

optimizer.max_iterations = 10

optimizer.optimize(dimensions = 2, function=cost)