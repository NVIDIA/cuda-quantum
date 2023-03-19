Creating Kernels Dynamically with the :code:`cudaq::kernel_builder`
-------------------------------------------------------------------
There may be use cases whereby one might need the ability to construct 
quantum kernels dynamically at runtime, as opposed to statically defining
kernel structs, lambdas, or functions. CUDA Quantum provides an abstraction called the 
:code:`cudaq::kernel_builder` for such cases. Imagine that you wanted to dynamically 
create a kernel with the following callable structure 

.. code-block:: cpp 

    struct kernel { 
      void operator()(double theta) __qpu__ { 
        cudaq::qreg<2> q;
        x(q[0]);
        ry(theta, q[1]);
        x<cudaq::ctrl>(q[1], q[0]);
      } 
    };

This can be expressed dynamically at runtime with the builder as follows 

.. code-block:: cpp 

    // Build a quantum kernel dynamically
    // Start by creating the cudaq::builder, the kernel argument types
    // should be provided here as template parameters.
    auto [ansatz, theta] = cudaq::make_kernel<double>();
    // Allocate some qubits
    auto q = ansatz.qalloc(2);
    // Build up the circuit, use the acquired parameter
    ansatz.x(q[0]);
    ansatz.ry(theta, q[1]);
    ansatz.x<cudaq::ctrl>(q[1], q[0]);
    
    // Can be used as input to any generic Kernel cudaq:: function
    using namespace cudaq::spin;
    cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                      .21829 * z(0) - 6.125 * z(1);
    auto exp_val = cudaq::observe(ansatz, h, /*theta*/ 0.59);

This builder pattern begins by creating the builder and any runtime
parameters required by the CUDA Quantum callable. The builder exposes an API 
for allocating qubits and applying quantum instructions. Once done 
constructing the kernel, the builder itself is callable, and can be 
passed as input to existing CUDA Quantum generic algorithm functions. 

The function parameters returned via the :code:`cudaq::make_kernel<T...>()` 
call can be modified algebraically, giving the programmer more control over the 
kernel construction. These parameters can be any arithmetic type or 
:code:`std::vector<T>` where T is any arithmetic type. The arithmetic
parameters can be multiplied by scalars, negated (unary negation), and summed
with scalars. Vector parameters can be indexed. Here's an example 

.. code-block:: cpp 

    // Create the kernel with signature void(std::vector<double>)
    auto [ansatz, thetas] = cudaq::make_kernel<std::vector<double>>();

    // Allocate some qubits
    auto q = ansatz.qalloc(3);

    // Build the kernel
    ansatz.x(q[0]);
    ansatz.ry(thetas[0], q[1]);
    ansatz.ry(thetas[1], q[2]);
    ansatz.x<cudaq::ctrl>(q[2], q[0]);
    ansatz.x<cudaq::ctrl>(q[0], q[1]);
    // Can do fancy arithmetic with Parameter types.
    ansatz.ry(-thetas[0], q[1]);
    // -or- ansatz_builder.ry(-1.0 * thetas[0], q[1]);
    // -or- ansatz_builder.ry(thetas[0] * -1.0, q[1]);
    // -or- ansatz_builder.ry(-1 * thetas[0], q[1]);
    ansatz.x<cudaq::ctrl>(q[0], q[1]);
    ansatz.x<cudaq::ctrl>(q[1], q[0]);

The :code:`cudaq::kernel_builder` internal builds up an MLIR representation of
the kernel. You can get a string representation of the MLIR code (in the Quake Dialect)
with the following call:

.. code-block:: cpp 
    
    auto quakeCode = ansatz.to_quake();
