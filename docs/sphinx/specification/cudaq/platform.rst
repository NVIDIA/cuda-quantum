
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
They take an optional :code:`QpuId` argument to specify the QPU of interest, or
default to the properties of the QPU in the current execution context.

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

      using QpuId = std::optional<std::size_t>;
      std::size_t get_num_qubits(QpuId qpu_id = std::nullopt) const;
      bool is_simulator(QpuId qpu_id = std::nullopt) const;
      bool is_remote(QpuId qpu_id = std::nullopt) const;
      bool is_emulated(QpuId qpu_id = std::nullopt) const;
      bool supports_conditional_feedback(QpuId qpu_id = std::nullopt) const;
      bool supports_explicit_measurements(QpuId qpu_id = std::nullopt) const;
      RemoteCapabilities get_remote_capabilities(QpuId qpuId = std::nullopt) const;
      std::string name() const;

    };
  }

CUDA-Q provides the following public functions to interact with the current
:code:`cudaq::quantum_platform`

.. code-block:: cpp

  namespace cudaq {
    quantum_platform &get_platform() ;
  }

