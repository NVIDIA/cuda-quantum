import cudaq
from cudaq import spin
import scipy

def objective_func(parameter_vector: list[float], hamiltonian, kernel, qubits_num):

    get_result = lambda parameter_vector: cudaq.observe(kernel, hamiltonian, qubits_num, parameter_vector).expectation() 
    cost = get_result(parameter_vector)
    print(f"<H> = {cost}")

    return cost

@cudaq.kernel(jit=True)
def ansatz(qubits: cudaq.qvector, thetas: list[float]):

    x(qubits[0])
    ry(thetas[0], qubits[1])
    x.ctrl(qubits[1], qubits[0])

@cudaq.kernel(jit=True)
def main_kernel(qubits_num: int, thetas: list[float]):

    qubits=cudaq.qvector(qubits_num)
    ansatz(qubits, thetas)

if __name__== "__main__":

    qubits_num: int= 2
    thetas: list[float]=[0.0]

    spin_ham = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    #result_vqe=scipy.optimize.minimize(objective_func,thetas,args=(spin_ham, main_kernel, qubits_num), method='L-BFGS-B', jac='3-point', tol=1e-8)
    result_vqe=scipy.optimize.minimize(objective_func,thetas,args=(spin_ham, main_kernel, qubits_num), method='COBYLA', tol=1e-8)
    
    print('Optimizer exited successfully: ',result_vqe.success)
    print(result_vqe.message)
    print('[Cudaq] Energy= ', result_vqe.fun)

