What is a CUDA-Q kernel?
-------------------------------

Quantum kernels are defined as functions that are executed on a quantum processing unit (QPU) or
a simulated QPU. They generalize quantum circuits and provide a new abstraction for quantum programming.
Quantum kernels can be combined with classical functions to create quantum-classical applications
that can be executed on a heterogeneous system of QPUs, GPUs, and CPUs to solve real-world problems.

**What’s the difference between a quantum kernel and a quantum circuit?**

Every quantum circuit is a kernel, but not every quantum kernel is a circuit. For instance, a quantum
kernel can be built up from other kernels, allowing us to interpret a large quantum program as a sequence
of subroutines or subcircuits.  

Moreover, since quantum kernels are functions, there is more expressibility available compared to a
standard quantum circuit. We can not only parameterize the kernel, but can also contain classical control
flow statements (`if`, `for`, `while`, etc.), and classical computations such as additions, multiplication, etc.
Conditional statements on quantum memory and qubit measurements can be included in quantum kernels to enable 
dynamic circuits and fast feedback, which are particularly useful for quantum error correction. To learn more about what
language constructs are supported within quantum kernels, take a look at the CUDA-Q 
:doc:`specification <../../specification/cudaq/kernels>`.

**Why doesn't calling a kernel behave like calling an ordinary function or method?**

A kernel does not execute on the same processor as the rest of your program: it runs on a QPU, or a
simulated QPU, which is a distinct processor with its own separate memory space (see the CUDA-Q
:doc:`machine model <../../specification/cudaq/machine_model>`). A reference into your program's own
memory has no meaning there - there is nothing on the QPU side for it to point to.

Because of this, calling a kernel is not quite the same as an ordinary function or method call, even
though the syntax looks identical. In Python, arguments to an ordinary function are passed by
reference, so a function can mutate the caller's own object; that does not apply to kernel arguments -
every argument is copied and passed to the kernel by value, so changes a kernel makes to its
arguments are never visible to the caller. Likewise, a kernel's return value is not written back into
an existing object: the value is produced in the QPU's own memory space and used to construct an
entirely new object back on the CPU side for the interpreter to use, not to update anything in place.
C++ kernels follow exactly the same pass-by-value rules, so this behavior is consistent across every
language CUDA-Q supports. See the :doc:`language specification <../../specification/cudaq/kernels>`
for the precise rules.

**How do I build and run a quantum kernel?**

Once a quantum kernel has been defined in a program, it may be called as a typical function, or can be executed
using the `sample`, `run` or `observe` primitives.

Let’s take a closer look at how to build and execute a quantum kernel with CUDA-Q.