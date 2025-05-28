import cudaq
from typing import Callable

# [Begin Kernel Composition Callable Python]
@cudaq.kernel()
def MyStatePrep_py(qubits : cudaq.qview):
    if qubits.size() > 0:
        h(qubits[0])
    print("MyStatePrep_py executed.")

@cudaq.kernel()
def MyGenericAlgorithm_py(statePrep : Callable[[cudaq.qview], None]):
    q = cudaq.qvector(10) # Example size
    statePrep(q)
    # ... other algorithm steps ...
    mz(q) # Measure
    print("MyGenericAlgorithm_py executed.")

if __name__ == "__main__":
    # Call the generic algorithm with the state prep kernel
    cudaq.sample(MyGenericAlgorithm_py, MyStatePrep_py)
# [End Kernel Composition Callable Python]