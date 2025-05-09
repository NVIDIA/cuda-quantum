import cudaq


cudaq.set_target("quantum_machines", 
                 url="http://host.docker.internal:8080", action="execute", executor="iqcc")

qubit_count = 3

@cudaq.kernel
def simplest():
    qvector = cudaq.qvector(qubit_count)


    for i in range(qubit_count):
        h(qvector[i])



cudaq.sample(simplest, shots_count=100)

#trans = cudaq.translate(three_qubit_ghz, format='openqasm2')
#print(trans)
