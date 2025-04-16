import cudaq


cudaq.set_target("quantum_machines")

qubit_count = 3

@cudaq.kernel
def three_qubit_ghz():
    qvector = cudaq.qvector(qubit_count)

    h(qvector[0])
    for i in range(1, 3):
        x.ctrl(qvector[0], qvector[i])

#cudaq.sample(three_qubit_ghz, shots_count=1000)

trans = cudaq.translate(three_qubit_ghz, format='openqasm2')
print(trans)