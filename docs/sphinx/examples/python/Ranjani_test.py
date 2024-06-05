import cudaq


print("start")

@cudaq.kernel
def NOTH(qubit: cudaq.qubit):
    #x(qubit)
    return

@cudaq.kernel

def OR_me():
    qubits = cudaq.qvector(3)
    #x(qubits[0])
    x(qubits[2])
    x(qubits[1])
    #cudaq.control(NOT,qubits[1],qubits[0])
    qubittemp=cudaq.qubit()
    x(qubittemp)
    cudaq.control(NOTH,qubits,qubittemp)
    #q4=OR(qubit3,qubittemp)


result = cudaq.sample(OR_me)
print(result)