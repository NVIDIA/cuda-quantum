import cudaq


@cudaq.kernel
def ghz(numQubits: int):
    qubits = cudaq.qvector(numQubits)
    h(qubits.front())
    for i, qubit in enumerate(qubits.front(numQubits - 1)):
        x.ctrl(qubit, qubits[i + 1])


counts = cudaq.sample(ghz, 10)
for bits, count in counts.items():
    print("Observed {} {} times.".format(bits, count))
