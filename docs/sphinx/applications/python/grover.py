import cudaq
from typing import Callable


@cudaq.kernel
def reflect(qubits: cudaq.qview):
    ctrls = qubits.front(qubits.size() - 1)
    last = qubits.back()
    cudaq.compute_action(lambda: (h(qubits), x(qubits)), lambda: z.ctrl(ctrls, last))


@cudaq.kernel
def oracle(q: cudaq.qview):
    z.ctrl(q[0], q[2])
    z.ctrl(q[1], q[2])


@cudaq.kernel
def grover(N: int, M: int, oracle: Callable[[cudaq.qview], None]):
    q = cudaq.qvector(N)
    h(q)
    for i in range(M):
        oracle(q)
        reflect(q)
    mz(q)


counts = cudaq.sample(grover, 3, 1, oracle)
assert len(counts) == 2
assert "101" in counts
assert "011" in counts
