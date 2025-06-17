import cudaq
import numpy as np

# [Begin Variable Capture Python]
# Python doesn't have direct struct member capture like C++ for kernels in the same way.
# Closures achieve similar effect.

i_host_py = 2
f_host_py = np.pi / 2.

@cudaq.kernel
def kernel_closure_py() -> int:
   # Use captured variables (Python closure)
   q = cudaq.qvector(i_host_py) # i_host_py from outer scope
   if i_host_py > 0:
       ry(f_host_py, q[0])    # f_host_py from outer scope
   mz(q)

   # Python integers are immutable. Rebinding 'i_host_py' here creates a local 'i_host_py'.
   # The outer 'i_host_py' is not affected.
   i_local_in_kernel = i_host_py # Make a new variable based on captured one
   i_local_in_kernel = 5
   print(f"Python closure kernel: captured i={i_host_py}, f={f_host_py}. Local modified i={i_local_in_kernel}")
   return i_local_in_kernel

if __name__ == "__main__":
    # The kernel returns an int.
    k_returned_py = kernel_closure_py() # Direct call returns the int value

    # Pass by value semantics for captured variables (effectively)
    # The original i_host_py in the outer scope is not changed by the kernel's internal assignment.
    print(f"Returned k_py = {k_returned_py}")
    print(f"Original i_host_py = {i_host_py}")
    assert k_returned_py != i_host_py # 5 != 2
    assert i_host_py == 2 # Original host variable unchanged
# [End Variable Capture Python]