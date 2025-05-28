import cudaq
from cudaq import h, mz 

# [Begin Variable Allocation Python]
@cudaq.kernel
def kernel_var_alloc_py():
   # Not Allowed in spirit of fixed-size for some backends
   # i_invalid = []
   # i_invalid.append(1)

   # Valid for Python, list comprehension creates fixed size list effectively
   i = [0 for _k_loop_var_ in range(5)] # _k_loop_var_ to indicate not used
   i[2] = 3

   f = [1., 2., 3.]

   k_int = 0
   k_int = i[2] # Use variable

   pi_val = 3.1415926
   pi_val += f[0] # Use variable

   # Minimal quantum operation
   q_dummy = cudaq.qvector(1)
   h(q_dummy[0])
   mz(q_dummy)
   print(f"Python kernel with variable allocations executed. i[2]={i[2]}, k_int={k_int}, pi_val={pi_val}")
# [End Variable Allocation Python]

# [Begin Variable Allocation Python Execution]
if __name__ == "__main__":
    cudaq.sample(kernel_var_alloc_py)
# [End Variable Allocation Python Execution]