Compiling and Executing CUDA Quantum Programs
---------------------------------------------
All CUDA Quantum programs must be compiled with the :code:`nvq++` compiler. This compiler 
orchestrates the mapping of quantum kernel expressions to the CUDA Quantum MLIR-based 
intermediate representation, quantum circuit optimization, placement, and routing, 
and lowering to LLVM-based representations (e.g. the QIR) for binary object 
creation and linking. The compiler supports a number of modes, simulators, and 
target quantum architectures, all of which can be configured with command line flags. 
All the usual compiler flags provided by compilers like :code:`clang++` and :code:`gcc` are also 
available for creating hybrid quantum-classical applications and libraries (e.g. 
linker flags and include header search paths). 

Given a CUDA Quantum source file named :code:`simple.cpp`, one can compile with :code:`nvq++` in
a similar manner as classical C++ compilers:

.. code-block:: console

    nvq++ simple.cpp -o simple.x (optional... -I /header/path -L /library/path -lfoo)
    ./simple.x

One can specify the target quantum architecture to target with the :code:`--target` flag:

.. code-block:: console 

    nvq++ simple.cpp --target nvidia
    ./a.out

The :code:`nvidia` target architecture will configure the executable to run 
on GPU-enabled simulation resources accelerated by cuQuantum. 

When targeting physical architectures, this simple compiler invocation orchestrates
a complex workflow that incorporates a number of tools (available in your
:code:`$CUDA_QUANTUM_PATH/bin` directory). 

1. Map CUDA Quantum C++ kernels to Quake MLIR code via Clang :code:`ASTConsumers`
2. Register all Quake kernel code with the runtime for quantum IR introspection
3. Rewrite the original CUDA Quantum C++ kernel entry-point function to invoke an internal runtime kernel
   launch function which targets the specified :code:`quantum_platform`
4. Lower to QIR and link, producing an executable or object code

For simulated architectures, the compiler relies on basic C++ compilation and linkage, and does not 
delegate to the MLIR compilation workflow.

You can see the steps taken to compile the CUDA Quantum code by adding the :code:`-v` verbose flag. 