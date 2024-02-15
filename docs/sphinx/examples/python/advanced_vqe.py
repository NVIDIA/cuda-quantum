import cudaq
from cudaq import spin
import scipy

@cudaq.kernel(jit=True)
def ansatz(qubits: cudaq.qvector, thetas: list[float]):

    x(qubits[0])
    ry(thetas[0], qubits[1])
    x.ctrl(qubits[1], qubits[0])

@cudaq.kernel(jit=True)
def main_kernel(qubits_num: int, thetas: list[float]):

    qubits=cudaq.qvector(qubits_num)
    ansatz(qubits, thetas)



qubits_num: int= 2
thetas: list[float]=[0.0]

spin_ham = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

optimizer = cudaq.optimizers.Adam()
gradient = cudaq.gradients.CentralDifference()

def objective_func(parameter_vector: list[float], hamiltonian=spin_ham, gradient=gradient, kernel=main_kernel, qubits_num=qubits_num):

    get_result = lambda parameter_vector: cudaq.observe(kernel, hamiltonian, qubits_num, parameter_vector).expectation() 
    #get_result = lambda parameter_vector: cudaq.observe(kernel, hamiltonian, qubits_num, parameter_vector, shots_count=100).expectation() 

    cost = get_result(parameter_vector)

    gradient_vector = gradient.compute(parameter_vector, get_result, cost)
    print(f"<H> = {cost}")

    return cost, gradient_vector

energy, parameter = optimizer.optimize(dimensions=1,function=objective_func)

print(f"\nminimized <H> = {round(energy,16)}")
print(f"optimal theta = {round(parameter[0],16)}")

