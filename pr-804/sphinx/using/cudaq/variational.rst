Variational Algorithms with CUDA Quantum
----------------------------------------
Variational algorithms in CUDA Quantum will typically leverage the 
:code:`cudaq::observe(...)` function in tandem with the :code:`cudaq::optimizer`.
Function optimization strategies are provided as specific sub-types of 
the :code:`cudaq::optimizer`. All optimizer implementations expose an 
:code:`optimize()` method that takes as input a callable (typically a lambda)
with the :code:`double(const std::vector<double>&, std::vector<double>&)`
signature for gradient-based optimizers, where the arguments represent the
function parameters at the current iteration and the modifiable gradient
vector reference, respectively. For gradient-free optimizers, the second 
argument, :code:`double(const std::vector<double>&)`, can be dropped.
Optimizers are defined in the header :code:`<cudaq/optimizers.h>` and 
gradients in :code:`<cudaq/gradients.h>`.

.. code-block:: cpp 
 
    // Define your parameterized ansatz
    struct ansatz {
      void operator()(double x0, double x1) __qpu__ {
        cudaq::qreg<3> q;
        x(q[0]);
        ry(x0, q[1]);
        ry(x1, q[2]);
        x<cudaq::ctrl>(q[2], q[0]);
        x<cudaq::ctrl>(q[0], q[1]);
        ry(-x0, q[1]);
        x<cudaq::ctrl>(q[0], q[1]);
        x<cudaq::ctrl>(q[1], q[0]);
      }  
    };

    cudaq::spin_op h = ...;

    // Create an Optimizer, here the COBYLA gradient-free optimizer
    // from the NLOpt library
    cudaq::optimizers::cobyla optimizer;

    // Optimize! Takes the number of objective function parameters and 
    // the objective function with the correct signature. 
    auto [opt_val, opt_params] = optimizer.optimize(2 /*Num Func Params*/,
       [&](const std::vector<double> &x) {
        // Map the incoming iteration parameters to the correct 
        // signature for your kernel as part of this observe call.
        // The kernel above takes 2 doubles, extract those from the parameter vector
        return cudaq::observe(ansatz{}, h, x[0], x[1]);
      });
    
The optimizers can leverage gradients that are computed from further CUDA Quantum kernel 
invocations. CUDA Quantum gradients require that kernel input parameters be mapped to a 
:code:`std::vector<double>`. CUDA Quantum kernels with signature :code:`void(std::vector<double>)`
are compatible with CUDA Quantum gradients out of the box, but those with non-default signature 
must provide a callable that maps kernel input arguments to a :code:`std::vector<double>`.
Here is an example 

.. code-block:: cpp 

    // Define your parameterized ansatz
    struct ansatz {
      void operator()(double x0, double x1) __qpu__ {
        cudaq::qreg<3> q;
        x(q[0]);
        ry(x0, q[1]);
        ry(x1, q[2]);
        x<cudaq::ctrl>(q[2], q[0]);
        x<cudaq::ctrl>(q[0], q[1]);
        ry(-x0, q[1]);
        x<cudaq::ctrl>(q[0], q[1]);
        x<cudaq::ctrl>(q[1], q[0]);
      }  
    };

    // Define an ArgMapper for the above kernel 
    // map std::vector<double> parameters to a 
    // tuple<double,double>, mirroring the (double,double) signature
    auto argMapper = [](std::vector<double> x) {
      return std::make_tuple(x[0], x[1]);
    };

    // Create a gradient-based Optimizer like L-BFGS
    cudaq::optimizers::lbfgs optimizer_lbfgs;

    // Create a gradient strategy. Needs the ansatz kernel and an 
    // ArgMapper if the kernel signature is non-default
    cudaq::gradients::parameter_shift gradient(ansatz{}, argMapper);

    auto [opt_val, opt_params] = optimizer_lbfgs.optimize(2 /*Num Func Params*/,
        [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
        // Compute the cost with the observe function, 
        // mapping the input vector to the kernel arguments
        auto cost = cudaq::observe(ansatz{}, h, x[0], x[1]);
        // Compute the gradient, needs the current parameters, the 
        // gradient reference (to modify), the spin_op, and the current cost.
        gradient.compute(x, grad_vec, h, cost);
        // Return to the optimizer
        return cost;
      });

CUDA Quantum provides the above code for the variational quantum eigensolver algorithm in 
a generic :code:`cudaq::` namespace function. The above snippets could be 
replaced with 

.. code-block:: cpp 

    // Gradient-free VQE
    cudaq::optimizers::cobyla optimizer;
    auto [opt_val, opt_params] =
        cudaq::vqe(ansatz{}, h, optimizer, /*n_params*/ 2);

    // Gradient-based VQE
    cudaq::optimizers::lbfgs anotherOptimizer;
    cudaq::gradients::parameter_shift gradient(ansatz{}, argMapper);
    auto [opt_val_2, opt_params_2] =
        cudaq::vqe(ansatz{}, gradient, h, anotherOptimizer, /*n_params*/ 2);
