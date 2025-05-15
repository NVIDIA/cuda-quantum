import cudaq


# simplest example of quantum machines target run:
# select a number of qubits (make sure you have this number of qubits available on your computer)
# and run a Hadamard on each of them.
# The default executor is mock, use executor name to run on another backend (real or simulator).
# Configure the address of the QOperator server in the url argument, and set the api_key
cudaq.set_target("quantum_machines", 
                 url="http://host.docker.internal:8080", api_key="1234567890", executor="mock")
qubit_count = 3

@cudaq.kernel
def simplest():
    qvector = cudaq.qvector(qubit_count)


    for i in range(qubit_count):
        h(qvector[i])



cudaq.sample(simplest, shots_count=100)

#trans = cudaq.translate(three_qubit_ghz, format='openqasm2')
#print(trans)
