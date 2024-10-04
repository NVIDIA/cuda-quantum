#!/usr/bin/env python
# coding: utf-8

# # Building Kernels
# 
# This section will cover the most basic CUDA-Q construct, a quantum kernel. Topics include, building kernels, initializing states, and applying gate operations.

# ### Defining Kernels
# 
# Kernels are the building blocks of quantum algorithms in CUDA-Q. A kernel is specified by using the following syntax. `cudaq.qubit` builds a register consisting of a single qubit, while `cudaq.qvector` builds a register of $N$ qubits.

# In[14]:


import cudaq


# In[15]:


@cudaq.kernel
def kernel():
    A = cudaq.qubit()
    B = cudaq.qvector(3)
    C = cudaq.qvector(5)


# Inputs to kernels are defined by specifying a parameter in the kernel definition along with the appropriate type. The kernel below takes an integer to define a register of N qubits.

# In[16]:


N = 2

@cudaq.kernel
def kernel(N: int):
    register = cudaq.qvector(N)


# ### Initializing states
# 
# It is often helpful to define an initial state for a kernel. There are a few ways to do this in CUDA-Q. Note, method 5 is particularly useful for cases where the state of one kernel is passed into a second kernel to prepare its initial state.
# 
# 1. Passing complex vectors as parameters
# 2. Capturing complex vectors
# 3. Precision-agnostic API
# 4. Define as CUDA-Q amplitudes
# 5. Pass in a state from another kernel

# In[17]:


# Passing complex vectors as parameters
c = [.707 +0j, 0-.707j]

@cudaq.kernel
def kernel(vec: list[complex]):
    q = cudaq.qubit(vec)


# Capturing complex vectors
c = [0.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)


# Precision-Agnostic API
import numpy as np

c = np.array([0.70710678 + 0j, 0., 0., 0.70710678], dtype=cudaq.complex())

@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)

# Define as CUDA-Q amplitudes
c = cudaq.amplitudes([0.70710678 + 0j, 0., 0., 0.70710678])

@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)

# Pass in a state from another kernel
c = [0.70710678 + 0j, 0., 0., 0.70710678]

@cudaq.kernel
def kernel_initial():
    q = cudaq.qvector(c)

state_to_pass = cudaq.get_state(kernel_initial)

@cudaq.kernel
def kernel(state: cudaq.State):
    q = cudaq.qvector(state)

kernel(state_to_pass)


# ### Applying Gates
# 
# 
# After a kernel is constructed, gates can be applied to start building out a quantum circuit. All the predefined gates in CUDA-Q can be found [here](https://nvidia.github.io/cuda-quantum/latest/api/default_ops.html#unitary-operations-on-qubits).
# 
# 
# Gates can be applied to all qubits in a register:

# In[18]:


@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    h(register)


# Or, to individual qubits in a register:

# In[19]:


@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    h(register[0])  # first qubit
    h(register[-1])  # last qubit


# ### Controlled Operations
# 
# Controlled operations are available for any gate and can be used by adding `.ctrl` to the end of any gate, followed by specification of the control qubit and the target qubit.

# In[20]:


@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    x.ctrl(register[0], register[1])  # CNOT gate applied with qubit 0 as control


# ### Multi-Controlled Operations
# 
# It is valid for more than one qubit to be used for multi-controlled gates. The control qubits are specified as a list.

# In[21]:


@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    x.ctrl([register[0], register[1]], register[2])  # X applied to qubit two controlled by qubit 0 and 1


# You can also call a controlled kernel within a kernel: 

# In[22]:


@cudaq.kernel
def x_kernel(qubit: cudaq.qubit):
    x(qubit)
    
# A kernel that will call `x_kernel` as a controlled operation.
@cudaq.kernel
def kernel():
    
    control_vector = cudaq.qvector(2)
    target = cudaq.qubit()
    
    x(control_vector)
    x(target)
    x(control_vector[1])
    cudaq.control(x_kernel, control_vector, target)

# The above is equivalent to: 

@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(3)
    x(qvector)
    x(qvector[1])
    x.ctrl([qvector[0], qvector[1]], qvector[2])
    mz(qvector)


results = cudaq.sample(kernel)
print(results)


# ### Adjoint Operations
# 
# The adjoint of a gate can be applied by appending the gate with the `adj` designation.

# In[23]:


@cudaq.kernel
def kernel():
    register = cudaq.qvector(10)
    t.adj(register[0])


# ### Custom Operations
# 
# Custom gate operations can be specified using `cudaq.register_operation`. A one-dimensional Numpy array specifies the unitary matrix to be applied. The entries of the array read from top to bottom through the rows.

# In[24]:


import numpy as np

cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

@cudaq.kernel
def kernel():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    custom_x(qubits[0])
    custom_x.ctrl(qubits[0], qubits[1])


# ### Building Kernels with Kernels
# 
# For many complex applications, it is helpful for a kernel to call another kernel to perform a specific subroutine. The example blow shows how `kernel_A` can be caled within `kernel_B` to perform CNOT operations.

# In[25]:


@cudaq.kernel
def kernel_A(qubit_0: cudaq.qubit, qubit_1: cudaq.qubit):
    x.ctrl(qubit_0, qubit_1)

@cudaq.kernel
def kernel_B():
    reg = cudaq.qvector(10)
    for i in range(5):
        kernel_A(reg[i], reg[i + 1])


# ### Parameterized Kernels
# 
# It is often useful to define parameterized circuit kernels which can be used for applications like VQE.

# In[26]:


@cudaq.kernel
def kernel(thetas: list[float]):
    qubits = cudaq.qvector(2)
    rx(thetas[0], qubits[0])
    ry(thetas[1], qubits[1])

thetas = [.024, .543]

kernel(thetas)

