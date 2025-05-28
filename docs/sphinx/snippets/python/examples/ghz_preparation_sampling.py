import cudaq

# [Begin GHZ State Python]
@cudaq.kernel
def ghz(numQubits:int):
    qubits = cudaq.qvector(numQubits)
    h(qubits[0]) # .front() is not standard Python list/qvector access
    for i in range(numQubits - 1): # Corrected loop and access
        x.ctrl(qubits[i], qubits[i + 1])
    mz(qubits) # Explicit mz for clarity, though sample adds it

# numQubits = 10 from original RST example
num_qubits_for_ghz_py = 4 # Example number of qubits
print(f"Sampling GHZ state for {num_qubits_for_ghz_py} qubits:")
counts = cudaq.sample(ghz, num_qubits_for_ghz_py)
for bits, count in counts.items(): # Corrected iteration
    print(f'Observed {bits} {count} times.')
# [End GHZ State Python]

if __name__ == "__main__":
    pass # Logic is at top level