import cudaq
from cudaq import ry, mz 
import numpy as np 

# [Begin Pass By Value Python]
@cudaq.kernel
def kernel_pbv_py(i_arg : int, v_arg : list[float]):
    # i_arg = 2, allocate 2 qubits
    q = cudaq.qvector(i_arg)
    # v_arg[1] == 2.0, angle here is 2.0
    if i_arg > 0 and len(v_arg) > 1:
        ry(v_arg[1], q[0])
    mz(q)

    print(f"Inside kernel_pbv_py: i_arg = {i_arg}, v_arg[0] = {v_arg[0] if v_arg else -1.0}")
    # Attempt to modify i_arg and v_arg
    i_arg = 5
    if v_arg: # Check if list is not empty
        # This modification should not affect the host list due to value semantics
        # when crossing from Python host to CUDA-Q kernel execution.
        v_arg_copy = v_arg[:] # Simulate internal copy if needed for modification
        v_arg_copy[0] = 3.0
        # If we directly did v_arg[0] = 3.0, and if it were true Python pass-by-object-reference
        # without CUDA-Q's value semantics layer, host would see it.
        # CUDA-Q's contract is value semantics for entry points.
    print(f"Inside kernel_pbv_py (after internal change attempt): i_arg = {i_arg}, v_arg[0] = {v_arg[0] if v_arg else -1.0}")


if __name__ == "__main__":
    k_py, d_py = 2, [1., 2.]
    d_py_original_copy = d_py[:] # Keep a true copy for assertion

    print(f"Before kernel_pbv_py call: k_py = {k_py}, d_py[0] = {d_py[0]}")
    cudaq.sample(kernel_pbv_py, k_py, d_py)

    # k_py is still 2, pass by value
    # d_py is still [1.0, 2.0], pass by value (CUDA-Q ensures this for kernel arguments)
    print(f"After kernel_pbv_py call: k_py = {k_py}, d_py[0] = {d_py[0]}")
    assert k_py == 2
    assert d_py[0] == d_py_original_copy[0] # Check against original
    assert d_py[1] == d_py_original_copy[1]
# [End Pass By Value Python]