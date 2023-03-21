Compiling and Executing CUDA Quantum Programs
---------------------------------------------
All CUDA Quantum programs must be compiled with the :code:`nvq++` compiler. This compiler 
orchestrates the mapping of quantum kernel expressions to the CUDA Quantum MLIR-based 
intermediate representation, quantum circuit optimization, placement, and routing, 
and lowering to LLVM-based representations (e.g. the QIR) for binary object 
creation and linking. The compiler supports a number of modes, platforms, and 
target QPU architectures, all of which can be configured with command line flags. 
All the usual compiler flags provided by compilers like clang++ and gcc are also 
available for creating hybrid quantum-classical applications and libraries (e.g. 
linker flags and include header search paths). 

Given a CUDA Quantum source file named :code:`simple.cpp`, one can compile with :code:`nvq++` in
a similar manner as classical C++ compilers:

.. code-block:: console

  nvq++ simple.cpp -o simple.x (optional... -I /header/path -L /library/path -lfoo)
  ./simple.x 

One can specify the QPU architecture to target with the :code:`--qpu` flag:

.. code-block:: console 

  nvq++ simple.cpp --qpu cuquantum 
  ./a.out 

The :code:`cuquantum` target QPU will configure the executable to run 
on GPU-enabled simulation resources accelerated by cuQuantum. 

This simple compiler invocation orchestrates a complex workflow that incorporates
a number of tools (available in your :code:`$CUDA_QUANTUM_PATH/bin` directory). 

1. Map CUDA Quantum C++ kernels to Quake MLIR code via Clang ASTConsumers
2. Register all Quake kernel code with the runtime for quantum IR introspection
3. Rewrite the original CUDA Quantum C++ kernel entry-point function to invoke an internal runtime kernel
   launch function which targets the specified :code:`quantum_platform`
4. Lower to QIR and link, producing an executable or object code
