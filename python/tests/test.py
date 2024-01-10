import cudaq
import numpy as np

cudaq.set_target("qpp-cpu")

kernel = cudaq.make_kernel()
qubits = kernel.qalloc(4)

# Should see the state vector in the |1>-state
# because that's what has been temporarily hard-coded 
# in `__quantum__rt__qubit_allocate_array`
print(cudaq.get_state(kernel))
# The measured bitstrings should be entirely in the |1> state,
# instead of the usual |0>.
print(cudaq.sample(kernel))