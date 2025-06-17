import cudaq

# [Begin Kernel Types Python]
@cudaq.kernel()
def my_first_entry_point_kernel_py(x : float):
   q = cudaq.qvector(1)
   ry(x, q[0])
   mz(q)
   print(f"Python lambda-like kernel executed with x = {x}")

@cudaq.kernel()
def my_second_entry_point_kernel_py(x : float, params : list[float]):
   q = cudaq.qvector(1)
   ry(x * params[0], q[0])
   mz(q)
   print(f"Python struct-like kernel executed with x = {x}, param0 = {params[0]}")

@cudaq.kernel()
def my_first_pure_device_kernel_py(qubits : cudaq.qview):
   h(qubits[0])
   print(f"Python pure device kernel executed on a qubit.")

@cudaq.kernel()
def caller_for_pure_device_py():
    q = cudaq.qvector(1)
    my_first_pure_device_kernel_py(q)
    mz(q)
# [End Kernel Types Python]

# [Begin Kernel Types Python Execution]
if __name__ == "__main__":
    my_first_entry_point_kernel_py(1.23)
    params_py = [0.5]
    my_second_entry_point_kernel_py(1.23, params_py)
    cudaq.sample(caller_for_pure_device_py)
# [End Kernel Types Python Execution]