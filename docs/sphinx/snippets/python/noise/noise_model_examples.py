import cudaq
import numpy as np

# Define a simple bit-flip channel for demonstration
# K0 = sqrt(1-p) * I, K1 = sqrt(p) * X
def create_py_bit_flip_channel(p: float) -> KrausChannel:
    if not (0.0 <= p <= 1.0):
        raise ValueError("Probability p must be between 0 and 1.")
    
    sqrt_1_minus_p = np.sqrt(1.0 - p)
    sqrt_p = np.sqrt(p)

    k0_matrix = np.array([[sqrt_1_minus_p, 0.0], [0.0, sqrt_1_minus_p]], dtype=complex)
    k1_matrix = np.array([[0.0, sqrt_p], [sqrt_p, 0.0]], dtype=complex)
    
    k0_op = cudaq.KrausOperator(k0_matrix)
    k1_op = cudaq.KrausOperator(k1_matrix)
    return cudaq.KrausChannel([k0_op, k1_op])

@cudaq.kernel
def ghz_py():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])
    # Apply some gates that will have noise
    z(q[0]) # Noise on specific qubit
    x(q[1]) # Noise on all qubits
    rx(1.23, q[2]) # Dynamic noise
    mz(q)

if __name__ == "__main__":
    py_bit_flip_channel = create_py_bit_flip_channel(0.1) # 10% bit flip

    noise_py = cudaq.NoiseModel() # Renamed from 'noise' in RST

    # [Begin PY AddChannelSpecific]
    # Add a noise channel to z gate on qubit 0
    noise_py.add_channel('z', [0], py_bit_flip_channel)
    # [End PY AddChannelSpecific]

    # [Begin PY AddChannelAllQubit]
    # Add a noise channel to x gate, regardless of qubit operands.
    noise_py.add_all_qubit_channel('x', py_bit_flip_channel)
    # [End PY AddChannelAllQubit]

    # [Begin PY AddChannelDynamic]
    # Noise channel callback function
    def noise_cb_py(qubits: list[int], params: list[float]) -> KrausChannel:
       print(f"Dynamic noise callback for rx on qubits: {qubits} with params: {params}")
       # Construct a channel based on specific operands and parameters
       # For simplicity, return the same bit-flip channel,
       # but could be dependent on qubits/params.
       p_dynamic = 0.05 # Default dynamic probability
       if params and params[0] > np.pi/2: # if angle > pi/2
           p_dynamic = 0.15 # higher error
       return create_py_bit_flip_channel(p_dynamic)
    
    # Add a dynamic noise channel to the 'rx' gate.
    noise_py.add_channel('rx', noise_cb_py)
    # [End PY AddChannelDynamic]

    cudaq.set_noise(noise_py)
    print("Python: Running GHZ with noise model.")
    counts_py = cudaq.sample(ghz_py)
    cudaq.unset_noise()

    print("Python: Counts with noise:")
    counts_py.dump()

    # For comparison, run without noise
    print("\nPython: Running GHZ without noise model.")
    ideal_counts_py = cudaq.sample(ghz_py)
    print("Python: Ideal counts:")
    ideal_counts_py.dump()