Quantum Algorithmic Primitives
******************************
**[1]** The general philosophy of the CUDA-Q specification is that quantum
device code should be encapsulated as stand-alone callable instances of generic
signature, and that operations or primitive algorithms targeting a quantum
coprocessor be implemented as adaptors on those callable instances. Adaptors, by
definition, are generic functions that take any quantum kernel as input along with
the runtime arguments for that kernel. Runtime arguments passed to adaptor functions
flow through the adaptor to the provided kernel. This pattern allows general pre-
and post-processing around concrete kernel execution.

.. _cudaq-sample-spec:

:code:`cudaq::sample`
-------------------------
**[1]** A common task for near-term quantum execution is to sample the state
of a given quantum circuit for a specified number of shots (circuit
executions). The result of this task is typically a mapping of observed
measurement bit strings to the number of times each was observed. This
is typically termed the counts dictionary in the community.

**[2]** The CUDA-Q model enables this functionality via template functions within the
:code:`cudaq` namespace with the following structure:

.. code-block:: cpp

    template <typename ReturnType>
    concept HasVoidReturnType = std::is_void_v<ReturnType>;

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

**[3]** This function takes as input a quantum kernel instance followed by the
concrete arguments at which the kernel should be invoked. CUDA-Q kernels
passed to this function must be entry-point kernels and return :code:`void`.

**[4]** Overloaded functions exist for specifying the number of shots to sample and the
noise model to apply.

**[5]** The function returns an instance of the :code:`cudaq::sample_result` type which encapsulates
the counts dictionary produced by the sampling task. Programmers can
extract the result information in the following manner:

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/algorithmic/sample_bell_state.cpp
     :language: cpp
     :start-after: [Begin Bell Kernel C++]
     :end-before: [End Bell Kernel C++]
  .. literalinclude:: /../snippets/cpp/algorithmic/sample_bell_state.cpp
     :language: cpp
     :start-after: [Begin Sample Bell C++]
     :end-before: [End Sample Bell C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/algorithmic/sample_bell_state.py
     :language: python
     :start-after: [Begin Bell Kernel Python]
     :end-before: [End Bell Kernel Python]
  .. literalinclude:: /../snippets/python/algorithmic/sample_bell_state.py
     :language: python
     :start-after: [Begin Sample Bell Python]
     :end-before: [End Sample Bell Python]


**[6]** CUDA-Q specifies the following structure for :code:`cudaq::sample_result`:

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

          double expectation(const std::string_view registerName == GlobalRegisterName);
          double probability(std::string_view bitString, const std::string_view registerName == GlobalRegisterName);
          std::size_t size(const std::string_view registerName == GlobalRegisterName);

          void dump();
          void clear();

          CountsDictionary::iterator begin();
          CountsDictionary::iterator end();
      };
    }

**[7]** By default the :code:`sample_result` type enables one to encode
measurement results from a quantum circuit sampling task. It keeps track of a
list of sample results, each one corresponding to a measurement action during
the sampling process and represented by a unique register name. It also tracks
a unique global register, which by default, contains the implicit sampling of
the state at the end of circuit execution. If the :code:`explicit_measurements`
sample option is enabled, the global register contains all measurements
concatenated together in the order the measurements occurred in the kernel.
The API gives fine-grain access to the measurement results for each register.
To illustrate this, observe

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/algorithmic/sample_explicit_measurements.cpp
     :language: cpp
     :start-after: [Begin Kernel C++]
     :end-before: [End Kernel C++]
  .. literalinclude:: /../snippets/cpp/algorithmic/sample_explicit_measurements.cpp
     :language: cpp
     :start-after: [Begin Sample C++]
     :end-before: [End Sample C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/algorithmic/sample_explicit_measurements.py
     :language: python
     :start-after: [Begin Kernel Python]
     :end-before: [End Kernel Python]
  .. literalinclude:: /../snippets/python/algorithmic/sample_explicit_measurements.py
     :language: python
     :start-after: [Begin Sample Python]
     :end-before: [End Sample Python]

should produce

.. code-block:: bash

    Default - no explicit measurements
    {
      __global__ : { 1:1000 }
      reg1 : { 0:506 1:494 }
    }

    Setting `explicit_measurements` option
    { 0:479 1:521 }

Here we see that we have measured a qubit in a uniform superposition to a
register named :code:`reg1`, and followed it with a reset and the application
of an NOT operation. By default the :code:`sample_result` returned for this
sampling tasks contains the default :code:`__global__` register as well as the
user specified :code:`reg1` register.

The contents of the :code:`__global__` register will depend on how your kernel
is written:

1. If no measurements appear in the kernel, then the :code:`__global__`
   register is formed with implicit measurements being added for *all* the
   qubits defined in the kernel, and the measurements all occur at the end of
   the kernel. This is not supported when sampling with the
   :code:`explicit_measurements` option; kernels executed with
   :code:`explicit_measurements` mode must contain measurements.
   The order of the bits in the bitstring corresponds to the qubit
   allocation order specified in the kernel.  That is - the :code:`[0]` element
   in the :code:`__global__` bitstring corresponds with the first declared qubit
   in the kernel. For example,

.. tab:: C++

   .. literalinclude:: /../snippets/cpp/algorithmic/sample_implicit_measurements_dump.cpp
      :language: cpp
      :start-after: [Begin Kernel C++]
      :end-before: [End Kernel C++]
   .. literalinclude:: /../snippets/cpp/algorithmic/sample_implicit_measurements_dump.cpp
      :language: cpp
      :start-after: [Begin Sample C++]
      :end-before: [End Sample C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/algorithmic/sample_implicit_measurements_dump.py
     :language: python
     :start-after: [Begin Kernel Python]
     :end-before: [End Kernel Python]
  .. literalinclude:: /../snippets/python/algorithmic/sample_implicit_measurements_dump.py
     :language: python
     :start-after: [Begin Sample Python]
     :end-before: [End Sample Python]

should produce

   .. code-block:: bash

       {
         __global__ : { 10:1000 }
       }

2. Conversely, if any measurements appear in the kernel, then only the measured
   qubits will appear in the :code:`__global__` register. Similar to #1, the
   bitstring corresponds to the qubit allocation order specified in the kernel.
   Also (again, similar to #1), the values of the sampled qubits always
   correspond to the values *at the end of the kernel execution*, unless the
   :code:`explicit_measurements` option is enabled. That is - if a qubit is
   measured in the middle of a kernel and subsequent operations change the state
   of the qubit, the qubit will be implicitly re-measured at the end of the
   kernel, and that re-measured value is the value that will appear in the
   :code:`__global__` register. If the sampling option :code:`explicit_measurements`
   is enabled, then no re-measurements occur, and the global register contains
   the concatenated measurements in the order they were executed in the kernel.

.. tab:: C++

   .. literalinclude:: /../snippets/cpp/algorithmic/sample_with_measurements_global_reg.cpp
      :language: cpp
      :start-after: [Begin Kernel C++]
      :end-before: [End Kernel C++]
   .. literalinclude:: /../snippets/cpp/algorithmic/sample_with_measurements_global_reg.cpp
      :language: cpp
      :start-after: [Begin Sample C++]
      :end-before: [End Sample C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/algorithmic/sample_with_measurements_global_reg.py
     :language: python
     :start-after: [Begin Kernel Python]
     :end-before: [End Kernel Python]
  .. literalinclude:: /../snippets/python/algorithmic/sample_with_measurements_global_reg.py
     :language: python
     :start-after: [Begin Sample Python]
     :end-before: [End Sample Python]

should produce

   .. code-block:: bash

       Default - no explicit measurements
       { 10:1000 }

       Setting `explicit_measurements` option
       { 01:1000 }

.. note::

  If you don't specify any measurements in your kernel and allow the :code:`nvq++`
  compiler to perform passes that introduce ancilla qubits into your kernel, it
  may be difficult to discern which qubits are the ancilla qubits vs which ones
  are your qubits. In this case, it is recommended that you provide explicit
  measurements in your kernel in order to only receive measurements from your
  qubits and silently discard the measurements from the ancillary qubits.

**[8]** The API exposed by the :code:`sample_result` data type allows one to extract
the information contained at a variety of levels and for each available
register name. One can get the number of times a bit string was observed via
:code:`sample_result::count`, extract a `std::unordered_map` representation via
:code:`sample_result::to_map`, get a new :code:`sample_result` instance over a subset of
measured qubits via :code:`sample_result::get_marginal`, and extract the
measurement data as it was produced sequentially (a vector of bit string observations
for each shot in the sampling process). One can also compute probabilities and expectation
values.

**[9]** There are specific requirements on input quantum kernels for the use of the
sample function which must be enforced by compiler implementations. The kernel
must be an entry-point kernel that returns :code:`void`.

**[10]** CUDA-Q also provides an asynchronous version of this function
(:code:`cudaq::sample_async`) which returns a :code:`sample_async_result`.

.. code-block:: cpp

    template<typename QuantumKernel, typename... Args>
    async_sample_result sample_async(const std::size_t qpu_id, QuantumKernel&& kernel, Args&&... args);

Programmers can asynchronously launch sampling tasks on any :code:`qpu_id`.

**[11]** The :code:`async_sample_result` wraps a :code:`std::future<sample_result>` and exposes the same
:code:`get()` functionality to extract the results after asynchronous execution.

**[12]** For remote QPU systems with long queue times, the :code:`async_sample_result` type encodes job ID
information and can be persisted to file and loaded from file at a later time. After loading from file,
and when remote queue jobs are completed, one can invoke :code:`get()` and the results will
be retrieved and returned.

:code:`cudaq::observe`
-------------------------
**[1]** A common task in variational algorithms is the computation of the expected
value of a given observable with respect to a parameterized quantum circuit
(:math:`\langle H \rangle(𝚹) = \langle \psi(𝚹)|H|\psi(𝚹) \rangle`).

**[2]** The :code:`cudaq::observe` function is provided to enable one to quickly compute
this expectation value via execution of the parameterized quantum circuit
with repeated measurements in the bases of the provided :code:`spin_op` terms. The
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

**[3]** :code:`cudaq::observe` takes as input an instantiated quantum kernel, the
:code:`cudaq::spin_op` whose expectation is requested, and the concrete
arguments used as input to the parameterized quantum kernel.

**[4]** :code:`cudaq::observe` returns an instance of the :code:`observe_result` type which can be implicitly
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
        double expectation();

        template <typename SpinOpType>
        double expectation(SpinOpType term);

        template <typename SpinOpType>
        sample_result counts(SpinOpType term);
        double id_coefficient()
        void dump();
    };

**[5]** The public API for :code:`observe_result` enables one to extract the
:code:`sample_result` data for each term in the provided :code:`spin_op`.
This return type can be used in the following way.

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/algorithmic/observe_result_usage.cpp
     :language: cpp
     :start-after: [Begin Kernel C++]
     :end-before: [End Kernel C++]
  .. literalinclude:: /../snippets/cpp/algorithmic/observe_result_usage.cpp
     :language: cpp
     :start-after: [Begin Observe Result Usage C++]
     :end-before: [End Observe Result Usage C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/algorithmic/observe_result_usage.py
     :language: python
     :start-after: [Begin Kernel Python]
     :end-before: [End Kernel Python]
  .. literalinclude:: /../snippets/python/algorithmic/observe_result_usage.py
     :language: python
     :start-after: [Begin Observe Result Usage Python]
     :end-before: [End Observe Result Usage Python]


Here is an example of the utility of the :code:`cudaq::observe` function:

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/algorithmic/observe_h2_ansatz_example.cpp
     :language: cpp
     :start-after: [Begin H2 Ansatz C++]
     :end-before: [End H2 Ansatz C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/algorithmic/observe_h2_ansatz_example.py
     :language: python
     :start-after: [Begin H2 Ansatz Python]
     :end-before: [End H2 Ansatz Python]


**[5]** There are specific requirements on input quantum kernels for the use of the
observe function which must be enforced by compiler implementations. The
kernel must be an entry-point kernel that does not contain any conditional
or measurement statements.

**[6]** By default on simulation backends, :code:`cudaq::observe` computes the true
analytic expectation value (i.e. without stochastic noise due to shots-based sampling).
If a specific shot count is provided then the returned expectation value will contain some
level of statistical noise. Overloaded :code:`observe` functions are provided to
specify the number of shots and/or specify the noise model to apply.

**[7]** CUDA-Q also provides an asynchronous version of this function
(:code:`cudaq::observe_async`) which returns a :code:`async_observe_result`.

.. code-block:: cpp

    template<typename QuantumKernel, typename... Args>
    async_observe_result observe_async(const std::size_t qpu_id, QuantumKernel&& kernel, cudaq::spin_op&, Args&&... args);

Programmers can asynchronously launch sampling tasks on any :code:`qpu_id`.

**[8]** For remote QPU systems with long queue times, the :code:`async_observe_result` type encodes job ID
information for each execution and can be persisted to file and loaded from file at a later time. After loading from file,
and when remote queue jobs are completed, one can invoke :code:`get()` and the results will
be retrieved and returned.

:code:`cudaq::optimizer` (deprecated, functionality moved to CUDA-Q libraries)
------------------------------------------------------------------------------------
The primary use case for :code:`cudaq::observe` is to leverage it as
the core of a broader objective function optimization workflow.
:code:`cudaq::observe` produces the expected value of a specified
:code:`spin_op` with respect to a given parameterized ansatz at a concrete
set of parameters, and often programmers will require an extremal value of that expected value
at a specific set of concrete parameters. This will directly require
abstractions for gradient-based and gradient-free optimization strategies.

The CUDA-Q model provides a :code:`cudaq::optimizer` data type that exposes
an :code:`optimize()` method that takes as input an
:code:`optimizable_function` to optimize and the number of independent
function dimensions. Implementations are free to implement this abstraction
in any way that is pertinent, but it is expected that most approaches will
enable optimization strategy extensibility. For example, programmers should
be able to instantiate a specific :code:`cudaq::optimizer` sub-type, thereby
dictating the underlying optimization algorithm in a type-safe manner.
Moreover, the optimizer should expose a public API of pertinent optimizer-specific
options that the programmer can customize.

CUDA-Q models the :code:`cudaq::optimizer` as follows:

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

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/algorithmic/optimizer_cobyla_example.cpp
     :language: cpp
     :start-after: [Begin COBYLA Example C++]
     :end-before: [End COBYLA Example C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/algorithmic/optimizer_cobyla_example.py
     :language: python
     :start-after: [Begin COBYLA Example Python]
     :end-before: [End COBYLA Example Python]

:code:`cudaq::gradient` (deprecated, functionality moved to CUDA-Q libraries)
-----------------------------------------------------------------------------------
Typical optimization use cases will require the computation of gradients for the specified
objective function. The gradient is a vector over all ansatz circuit
parameters :math:`∂H(𝚹) / ∂𝚹_i`. There are a number of potential strategies for
computing this gradient vector, but most require additional evaluations
of the ansatz circuit on the quantum processor.

To enable true extensibility in gradient strategies, CUDA-Q programmers can
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
                  double exp_h) override { /* ... */ } // Implementation details omitted
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

The overall CUDA-Q workflow for leveraging the :code:`cudaq::optimizer`
will work as follows (here we demonstrate with an ansatz without the
default :code:`std::vector<double>` signature):

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/algorithmic/optimizer_lbfgs_gradient_example.cpp
     :language: cpp
     :start-after: [Begin LBFGS Example C++]
     :end-before: [End LBFGS Example C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/algorithmic/optimizer_lbfgs_gradient_example.py
     :language: python
     :start-after: [Begin LBFGS Example Python]
     :end-before: [End LBFGS Example Python]