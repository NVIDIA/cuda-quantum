import cudaq
from dataclasses import dataclass
import numpy as np
from typing import List 

# [Begin Allowed Types Python]
@dataclass
class MySimpleStructPy:
    i : int
    j : int
    # Python lists are dynamic, so for kernel, fixed size is implied by usage
    # For dataclass, it's just a Python list.
    angles: list[float]


@cudaq.kernel
def kernel_allowed_types_py(N : int, flag : bool, angle : float, layers : list[int],
            parameters : list[float], recursiveList : list[list[float]],
            var : MySimpleStructPy):
    q = cudaq.qvector(1)
    if flag and N > 0 and layers and parameters and recursiveList and var.angles:
         ry(angle + parameters[0] + recursiveList[0][0] + var.angles[0], q[0])
    mz(q)
    print("Python kernel with allowed types executed.")

@cudaq.kernel
def kernelThatReturns_py() -> float:
    q = cudaq.qvector(1)
    h(q[0])
    mz(q) # Measurement
    return np.pi / 2.0
# [End Allowed Types Python]

# [Begin Allowed Types Python Execution]
if __name__ == "__main__":
    s_var_py = MySimpleStructPy(i=1, j=2, angles=[0.1, 0.2])
    l_py = [0,1]
    p_py = [0.5]
    rv_py = [[0.3]]
    kernel_allowed_types_py(1, True, 0.5, l_py, p_py, rv_py, s_var_py)

    # For a kernel returning a float, observe or sample might be used.
    # `cudaq.observe` is suitable if the float is an expectation value.
    # `cudaq.sample` will give counts; the float return isn't directly in `sample_result`
    # unless it's a measured bitstring interpreted as a float (uncommon).
    # The RST implies a direct return.
    # If the kernel is called and its Python return value is used:
    result_val_py = kernelThatReturns_py() # This executes the kernel
    print(f"Python kernel that returns: {result_val_py}")
# [End Allowed Types Python Execution]