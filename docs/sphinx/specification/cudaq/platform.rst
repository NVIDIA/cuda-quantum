
Quantum Platform
****************

**[1]** CUDA-Q provides an abstraction describing the underlying quantum computing resource(s). 
The underlying quantum platform can contain one or many quantum processing units (QPUs) each with 
its own qubit connectivity and noise configuration.

**[2]** CUDA-Q defines a :code:`cudaq::quantum_platform` in an effort to expose relevant system 
information and enable asynchronous quantum kernel invocations.

**[3]** The :code:`cudaq::quantum_platform` provides an API for querying the number of available 
quantum processing units (QPUs), with each QPU assigned a logical integer index (:code:`{0,1,2,...}`). 

**[4]** The properties of the QPUs on the platform are exposed by a collection
of functions such as :code:`get_num_qubits`, :code:`is_simulator`, :code:`is_remote`, :code:`is_emulated`, etc.
They take an optional :code:`qpu_id` argument (defaults to 0) to specify the QPU of interest.

**[5]** The ID of the desired QPU can be specified as argument when invoking
kernel functions such as :code:`cudaq::sample_async`, :code:`cudaq::run_async`, etc.

The :code:`cudaq::quantum_platform`  should take the following structure

.. code-block:: cpp

  namespace cudaq {
    class quantum_platform {
    public:
      quantum_platform();
      ~quantum_platform();
 
      using QubitEdge = std::pair<std::size_t, std::size_t>;
      using QubitConnectivity = std::vector<QubitEdge>;
      std::optional<QubitConnectivity> connectivity();

      std::size_t num_qpus() const;

      std::size_t get_num_qubits(std::size_t qpu_id = 0) const;
      bool is_simulator(std::size_t qpu_id = 0) const;
      bool is_remote(std::size_t qpu_id = 0) const;
      bool is_emulated(std::size_t qpu_id = 0) const;
      bool supports_conditional_feedback(std::size_t qpu_id = 0) const;
      bool supports_explicit_measurements(std::size_t qpu_id = 0) const;
      RemoteCapabilities get_remote_capabilities(std::size_t qpu_id = 0) const;
      std::string name() const;

    };
  }

CUDA-Q provides the following public functions to interact with the current
:code:`cudaq::quantum_platform`

.. code-block:: cpp

  namespace cudaq {
    quantum_platform &get_platform() ;
  }

