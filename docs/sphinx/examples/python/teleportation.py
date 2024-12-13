import cudaq

# Alice wants to send her data to Bob through teleportation through below simple quantum teleport protocol

@cudaq.kernel
def create_bell_pair(q1: cudaq.qubit, q2: cudaq.qubit):
    h(q1)
    cx(q1, q2)

@cudaq.kernel
def alter_alice_data(q0: cudaq.qubit):
    # Alice's data is always having state of 1
    x(q0)

@cudaq.kernel
def protocol():
    # initiate a qubit system of 3 qubits where
    # qubits[0] is Alice's data to be teleported
    # qubits[1] (Alice's state) and qubits[2] (Bob's state) will be entangled
    qubits = cudaq.qvector(3)
    create_bell_pair(qubits[1], qubits[2])
    # if comment below line, Alice's data is always having state of 0
    alter_alice_data(qubits[0])

    # bell measurement
    cx(qubits[0], qubits[1])
    h(qubits[0])
    mz(qubits[0])
    mz(qubits[1])

    # recovers Alice's data on Bob's state
    cx(qubits[1], qubits[2])
    cz(qubits[0], qubits[2])
    mz(qubits[2])

    mz(qubits)

print(cudaq.draw(protocol))
'''teleportation circuit:
     ╭───╮          ╭───╮     
q0 : ┤ x ├───────●──┤ h ├──●──
     ├───┤     ╭─┴─╮╰───╯  │  
q1 : ┤ h ├──●──┤ x ├──●────┼──
     ╰───╯╭─┴─╮╰───╯╭─┴─╮╭─┴─╮
q2 : ─────┤ x ├─────┤ x ├┤ z ├
          ╰───╯     ╰───╯╰───╯
'''
# if you don't choose to alter Alice's data, the x-gate on q0 will not be applied

result = cudaq.sample(protocol, shots_count=1024)
print(result)
# sample result: { 001:242 101:263 011:250 111:269 }
# teleportation protocol is successfully implemented since all Bob's state is always 1 which is expected
# if you don't choose to alter Alice's data in the protocol, all Bob's state will be 0:
# { 000:257 010:273 100:255 110:239 }