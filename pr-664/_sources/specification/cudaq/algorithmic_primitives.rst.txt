Quantum Algorithmic Primitives
******************************
The CUDA Quantum model provides the implementation of common quantum algorithmic
primitives within the :code:`cudaq::` namespace. Here we enumerate available
function calls:

:code:`cudaq::sample`
-------------------------
A common task for near-term quantum execution is to sample the state
of a given quantum circuit for a specified number of shots (circuit
executions). The result of this task is typically a mapping of observed
measurement bit strings to the number of times each was observed. This
is typically termed the counts dictionary in the community. 

The CUDA Quantum model enables this functionality via template functions within the
:code:`cudaq` namespace with the following structure:

.. code-block:: cpp

    // Kernel only
    template<typename QuantumKernel, typename... Args>
      requires HasVoidReturnType<QuantumKernel, Args...>
    sample_result sample(QuantumKernel&& kernel, Args&&... args);

    // Specify shots
    template<typename QuantumKernel, typename... Args>
      requires HasVoidReturnType<QuantumKernel, Args...>
    sample_result sample(std::size_t shots, QuantumKernel&& kernel, Args&&... args);

    // Specify sample options (including shots and noise model)
    template<typename QuantumKernel, typename... Args>
      requires HasVoidReturnType<QuantumKernel, Args...>
    sample_result sample(const sample_options &options,
                         QuantumKernel&& kernel, Args&&... args);

This function takes as input a quantum kernel instance followed by the
concrete arguments at which the kernel should be invoked. CUDA Quantum kernels 
passed to this function must be entry-point kernels and return :code:`void`. 

Overloaded functions exists for specifying the number of shots to sample and the
noise model to apply.

The function returns an instance of the :code:`cudaq::sample_result` type which encapsulates
the counts dictionary produced by the sampling task. Programmers can
extract the result information in the following manner: 

.. code-block:: cpp

    auto bell = []() __qpu__ { ... };
    auto counts = cudaq::sample(bell)
 
    // Print to standard out
    counts.dump();
 
    // Fine-grained access to the bits and counts
    for (auto& [bits, count] : counts) {
      printf("Observed: %s, %lu\n", bits, count);
    }

CUDA Quantum specifies the following structure for :code:`cudaq::sample_result`:

.. code-block:: cpp 

    namespace cudaq {
      using CountsDictionary = std::unordered_map<std::string, std::size_t>;
      inline static const std::string GlobalRegisterName = "__global__";
      class sample_result {
        public:
          sample_result() = default;
          sample_result(const sample_result &);
          ~sample_result();
          
          std::vector<std::string> register_names();
          
          std::size_t count(std::string_view bitString,
                      const std::string_view registerName = GlobalRegisterName);
          
          std::vector<std::string>
          sequential_data(const std::string_view registerName = GlobalRegisterName);
          
          CountsDictionary
          to_map(const std::string_view registerName = GlobalRegisterName);
          
          sample_result
          get_marginal(const std::vector<std::size_t> &&marginalIndices,
                 const std::string_view registerName = GlobalRegisterName);

          double exp_val_z(const std::string_view registerName == GlobalRegisterName);
          double probability(std::string_view bitString, const std::string_view registerName == GlobalRegisterName);
          std::size_t size(const std::string_view registerName == GlobalRegisterName);
          
          void dump();
          void clear();

          CountsDictionary::iterator begin();
          CountsDictionary::iterator end();
      };
    }

The :code:`sample_result` type enables one to encode measurement results from a 
quantum circuit sampling task. It keeps track of a list of sample results, each 
one corresponding to a measurement action during the sampling process and represented 
by a unique register name. It also tracks a unique global register, the implicit sampling 
of the state at the end of circuit execution. The API gives fine-grain access 
to the measurement results for each register. To illustrate this, observe 

.. code-block:: cpp

    auto kernel = []() __qpu__ {
      cudaq::qubit q;
      h(q);
      auto reg1 = mz(q);
      reset (q);
      x(q);
    };
    cudaq::sample(kernel).dump();

should produce 

.. code-block:: bash 

    { 
      __global__ : { 1:1000 }
      reg1 : { 0:501 1:499 }
    }

Here we see that we have measured a qubit in a uniform superposition to a 
register named :code:`reg1`, and followed it with a reset and the application 
of an NOT operation. The :code:`sample_result` returned for this sampling 
tasks contains the default :code:`__global__` register as well as the user 
specified :code:`reg1` register. 

The API exposed by the :code:`sample_result` data type allows one to extract
the information contained at a variety of levels and for each available 
register name. One can get the number of times a bit string was observed via 
:code:`sample_result::count`, extract a `std::unordered_map` representation via 
:code:`sample_result::to_map`, get a new :code:`sample_result` instance over a subset of 
measured qubits via :code:`sample_result::get_marginal`, and extract the 
measurement data as it was produced sequentially (a vector of bit string observations 
for each shot in the sampling process). One can also compute probabilities and expectation 
values. 

There are specific requirements on input quantum kernels for the use of the
sample function which must be enforced by compiler implementations.
The kernel must be an entry-point kernel that returns :code:`void`.

CUDA Quantum also provides an asynchronous version of this function 
(:code:`cudaq::sample_async`) which returns a 
:code:`sample_async_result`. 

.. code-block:: cpp 

    template<typename QuantumKernel, typename... Args>
    async_sample_result sample_async(const std::size_t qpu_id, QuantumKernel&& kernel, Args&&... args);

Programmers can asynchronously launch sampling tasks on any :code:`qpu_id`. 

The :code:`async_sample_result` wraps a :code:`std::future<sample_result>` and exposes the same 
:code:`get()` functionality to extract the results after asynchronous execution. 

For remote QPU systems with long queue times, the :code:`async_sample_result` type encodes job ID 
information and can be persisted to file and loaded from file at a later time. After loading from file, 
and when remote queue jobs are completed, one can invoke :code:`get()` and the results will 
be retrieved and returned. 

:code:`cudaq::observe`
-------------------------
A common task in variational algorithms is the computation of the expected
value of a given observable with respect to a parameterized quantum circuit
(:math:`\langle H \rangle(𝚹) = \langle \psi(𝚹)|H|\psi(𝚹) \rangle`). 

The :code:`cudaq::observe` function is provided to enable one to quickly compute
this expectation value via execution of the parameterized quantum circuit
with repeated measurements in the bases of the provided spin_op terms. The
function has the following signature:

.. code-block:: cpp
  
    // Kernel only
    template<typename QuantumKernel, typename... Args>
    observe_result observe(QuantumKernel&&, cudaq::spin_op&, Args&&... args);
    
    // Specify shots
    template<typename QuantumKernel, typename... Args>
    observe_result observe(std::size_t shots, QuantumKernel&&, cudaq::spin_op&, Args&&... args);

    // Specify sample options (including shots and noise model)
    template<typename QuantumKernel, typename... Args>
    observe_result observe(const cudaq::observe_options &options,
                          QuantumKernel&&, cudaq::spin_op&, Args&&... args);

This function takes as input an instantiated quantum kernel, the
:code:`cudaq::spin_op` whose expectation is requested, and the concrete
arguments used as input to the parameterized quantum kernel. This function
returns an instance of the :code:`observe_result` type which can be implicitly 
converted to a :code:`double` expectation value, but also retains all data directly
generated and used as part of that expectation value computation. The 
:code:`observe_result` takes on the following form:

.. code-block:: cpp

    class observe_result {
      public:
        observe_result(double &e, spin_op &H);
        observe_result(double &e, spin_op &H, MeasureCounts counts);

        sample_results raw_data() { return data; };
        operator double();
        double exp_val_z();
        
        template <typename SpinOpType>
        double exp_val_z(SpinOpType term);

        template <typename SpinOpType>
        sample_result counts(SpinOpType term);
        double id_coefficient() 
        void dump();
    };

The public API for :code:`observe_result` enables one to extract the 
:code:`sample_result` data for each term in the provided :code:`spin_op`. 
This return type can be used in the following way.

.. code-block:: cpp 

    // I only care about the expected value, discard 
    // the fine-grain data produced
    double expVal = cudaq::observe(kernel, spinOp, args...);

    // I require the result with all generated data 
    auto result = cudaq::observe(kernel, spinOp, args...);
    auto expVal = result.exp_val_z();
    auto X0X1Exp = result.exp_val_z(x(0)*x(1));
    auto X0X1Data = result.counts(x(0)*x(1));
    result.dump();

Here is an example of the utility of the :code:`cudaq::observe` function:

.. code-block:: cpp

    struct ansatz {
      auto operator()(double theta) __qpu__ {
        cudaq::qreg q(2);
        x(q[0]);
        ry(theta, q[1]);
        x<cudaq::ctrl>(q[1], q[0]);
      }
    };
  
    int main() {
      using namespace cudaq::spin; // make it easier to use pauli X,Y,Z below
  
      spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                  .21829 * z(0) - 6.125 * z(1);
  
      double energy = cudaq::observe(ansatz{}, h, .59);
      printf("Energy is %lf\n", energy); 
      return 0;
    }

There are specific requirements on input quantum kernels for the use of the
observe function which must be enforced by compiler implementations. The
kernel must be an entry-point kernel that does not contain any conditional
or measurement statements.

By default on simulation backends, :code:`cudaq::observe` computes the true
analytic expectation value (i.e. without stochastic noise due to shots-based sampling). 
If a specific shot count is provided then the returned expectation value will contain some 
level of statistical noise. Overloaded :code:`observe` functions are provided to 
specify the number of shots and/or specify the noise model to apply.

CUDA Quantum also provides an asynchronous version of this function 
(:code:`cudaq::observe_async`) which returns a :code:`async_observe_result`. 

.. code-block:: cpp 

    template<typename QuantumKernel, typename... Args>
    async_observe_result observe_async(const std::size_t qpu_id, QuantumKernel&& kernel, cudaq::spin_op&, Args&&... args);

Programmers can asynchronously launch sampling tasks on any :code:`qpu_id`. 

For remote QPU systems with long queue times, the :code:`async_observe_result` type encodes job ID 
information for each execution and can be persisted to file and loaded from file at a later time. After loading from file, 
and when remote queue jobs are completed, one can invoke :code:`get()` and the results will 
be retrieved and returned. 

:code:`cudaq::optimizer`
-------------------------
The primary use case for :code:`cudaq::observe` is to leverage it as
the core of a broader objective function optimization workflow. 
:code:`cudaq::observe` produces the expected value of a specified 
:code:`spin_op` with respect to a given parameterized ansatz at a concrete
set of parameters, and often programmers will require an extremal value of that expected value 
at a specific set of concrete parameters. This will directly require
abstractions for gradient-based and gradient-free optimization strategies. 

The CUDA Quantum model provides a :code:`cudaq::optimizer` data type that exposes
an :code:`optimize()` method that takes as input an 
:code:`optimizable_function` to optimize and the number of independent
function dimensions. Implementations are free to implement this abstraction
in any way that is pertinent, but it is expected that most approaches will
enable optimization strategy extensibility. For example, programmers should
be able to instantiate a specific :code:`cudaq::optimizer` sub-type, thereby 
dictating the underlying optimization algorithm in a type-safe manner. 
Moreover, the optimizer should expose a public API of pertinent optimizer-specific 
options that the programmer can customize.

CUDA Quantum models the :code:`cudaq::optimizer` as follows:

.. code-block:: cpp 

    namespace cudaq {
      // Encode the optimal value and optimal parameters
      using optimization_result = std::tuple<double, std::vector<double>>;
      // Initialized with user specified callable of a specific signature
      // Clients can query if the function computes gradients or not
      class optimizable_function {
        public:
          template<typename Callable>
          optimizable_function(Callable&&);
          bool providesGradients() { return _providesGradients; }
          double operator()(const std::vector<double> &x, std::vector<double> &dx);
      };
      class optimizer {
        public:
          virtual bool requiresGradients() = 0;
          virtual optimization_result optimize(const int dimensions,
                                              optimizable_function&& opt_function) = 0;
      }; 
    }

Here, :code:`optimization_result` should encode the optimal value and optimal
parameters achieved during the optimization workflow
(i.e. a :code:`tuple<double, std::vector<double>>`). The optimize method takes
as input the number of parameters (or dimensions of the objective function),
and a function-like object (i.e. :code:`std::function` or a lambda, something 
:code:`optimizable_function` can be constructed from) that takes a 
:code:`const std::vector<double>&` and :code:`std::vector<double>&` for the
function input parameters and gradient vector, respectively. The objective
function must return a double representing the scalar cost for the
objective function (e.g. the expected value from :code:`cudaq::observe()`).  

Here is an example of how the :code:`cudaq::optimizer` is intended to be used: 

.. code-block:: cpp 

    auto ansatz = [](double theta, double phi) __qpu__ {...};
    cudaq::spin_op H = ... ;
  
    cudaq::optimizers::cobyla optimizer;
    optimizer.max_eval = 200;
  
    auto [opt_energy, opt_params] = optimizer.optimize(
          2, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
            return cudaq::observe(ansatz, H, x[0], x[1]);
          });

:code:`cudaq::gradient`
-------------------------
Typical optimization use cases will require the computation of gradients for the specified
objective function. The gradient is a vector over all ansatz circuit
parameters :math:`∂H(𝚹) / ∂𝚹_i`. There are a number of potential strategies for
computing this gradient vector, but most require additional evaluations
of the ansatz circuit on the quantum processor. 

To enable true extensibility in gradient strategies, CUDA Quantum programmers can
instantiate custom sub-types of the :code:`cudaq::gradient` type. The :code:`cudaq::gradient`
type defines a :code:`compute(...)` method that takes a mutable reference to the 
current gradient vector and is free to update that vector in a strategy-specific way. 
The method also takes the current evaluation parameter vector, the :code:`cudaq::spin_op` used 
in the current variational task, and the computed expected value at the given parameters. 
The gradient strategy type takes the following form:

.. code-block:: cpp

    namespace cudaq {
      class gradient {
        public:
          gradient(std::function<void(std::vector<double>)> &&kernel);
  
          template <typename QuantumKernel, typename ArgsMapper>
          gradient(QuantumKernel &&kernel, ArgsMapper &&argsMapper);
  
          virtual void compute(std::vector<double>& x, std::vector<double> &dx,
                            spin_op& h, double exp_h) = 0;
          
          virtual std::vector<double>
          compute(const std::vector<double> &x,
                  std::function<double(std::vector<double>)> &func) = 0;

      };
  
      // gradient is intended for subclassing
      class central_difference : public gradient {
        public:
          void compute(std::vector<double>& x, std::vector<double> &dx, spin_op& h,
                  double exp_h) override { ... }
      };
    }

The :code:`compute` function can make use of the quantum kernel parameterized ansatz, the 
:code:`spin_op` for which the expected value is being computed, the
pre-computed expected value at the current iteration's parameter, and the
concrete arguments for the given quantum kernel at this iteration.

A non-trivial aspect of the computation of gradients (in an extensible manner)
is that we model the gradient as a derivative over concrete parameters for the
circuit ansatz represented as a :code:`std::vector<double>` when the actual
quantum kernel may be defined with general variadic :code:`Args...` types.
To address this issue, programmers can provide a default translation
mechanism for mapping common quantum kernel ansatz functional expressions to a :code:`vector<double>` representation - the 
:code:`ArgsMapper` callable template type. This type must implement the 
:code:`std::tuple<Args...>(std::vector<double>&)` callable concept. 

The overall CUDA Quantum workflow for leveraging the :code:`cudaq::optimizer`
will work as follows (here we demonstrate with an ansatz without the
default :code:`std::vector<double>` signature):

.. code-block:: cpp

    auto deuteron_n3_ansatz = [](double x0, double x1) __qpu__ {
      cudaq::qreg q(3);
      x(q[0]);
      ry(x0, q[1]);
      ry(x1, q[2]);
      x<cudaq::ctrl>(q[2], q[0]);
      x<vctrl>(q[0], q[1]);
      ry(-x0, q[1]);
      x<cudaq::ctrl>(q[0], q[1]);
      x<cudaq::ctrl>(q[1], q[0]);
    };

    cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
              .21829 * z(0) - 6.125 * z(1);
    cudaq::spin_op h3 = h + 9.625 - 9.625 * z(2) - 3.913119 * x(1) * x(2) -
                3.913119 * y(1) * y(2);

    // The above ansatz takes 2 doubles, not a single std::vector<double>, which 
    // the gradient type is expecting. So we must provide an ArgsMapper callable type
    auto argsMapper = [](std::vector<double> x) {return std::make_tuple(x[0],x[1]);};

    // Create the gradient strategy
    cudaq::gradients::central_difference gradient(deuteron_n3_ansatz, argsMapper);

    // Create the L-BFGS optimizer, requires gradients
    cudaq::optimizers::lbfgs optimizer;

    // Run the optimization routine. 
    auto [min_val, opt_params] = optimizer.optimize(
        2, [&](const std::vector<double>& x, std::vector<double>& grad_vec) {
          // Compute the cost, here its an energy
          auto cost = cudaq::observe(deuteron_n3_ansatz, h3, x);
          
          // Compute the gradient, results written to the grad_vec reference
          gradient.compute(x, grad_vec, h3, cost);

          // Return the cost to the optimizer
          return cost;
        });

    // Print the results
    printf("Optimizer found %lf at [%lf,%lf]\n", min_val, opt_params[0], opt_params[1]);

