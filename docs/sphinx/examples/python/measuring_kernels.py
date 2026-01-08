# [Begin Docs]
import cudaq
# [End Docs]


# [Begin Sample1]
@cudaq.kernel
def kernel():
    qubits = cudaq.qvector(2)
    mz(qubits)


# [End Sample1]


# [Begin Sample2]
@cudaq.kernel
def kernel():
    qubits_a = cudaq.qvector(2)
    qubit_b = cudaq.qubit()
    mz(qubits_a)
    mx(qubit_b)


# [End Sample2]


# [Begin Run0]
@cudaq.kernel
def kernel() -> list[bool]:
    q = cudaq.qvector(2)

    h(q[0])
    b0 = mz(q[0])
    reset(q[0])
    x(q[0])

    if b0:
        h(q[1])

    return mz(q)


from collections import Counter

results = cudaq.run(kernel, shots_count=1000)
# Convert results to bitstrings and count
bitstring_counts = Counter(
    ''.join('1' if bit else '0' for bit in result) for result in results)

print(f"Bitstring counts: {dict(bitstring_counts)}")

# [End Run0]

''' [Begin Run1]
Bitstring counts: {'11': 247, '10': 753}
 [End Run1] '''
