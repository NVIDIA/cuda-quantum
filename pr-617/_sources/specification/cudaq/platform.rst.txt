
Quantum Platform
****************
CUDA Quantum provides an abstraction describing the underlying quantum computing
resource(s). The underlying quantum platform can contain one or many quantum
processing units (QPUs) each with its own qubit connectivity and noise
configuration. 

CUDA Quantum defines a :code:`cudaq::quantum_platform` in an effort to expose 
relevant system information and enable asynchronous quantum kernel invocations. 

The :code:`cudaq::quantum_platform` provides an API for querying the number
of available quantum processing units (QPUs), with each QPU assigned a
logical integer index (:code:`{0,1,2,...}`). Programmers can specify the
ID of the desired QPU and all subsequent CUDA Quantum kernel executions will
target that QPU. 

The :code:`cudaq::quantum_platform` should take the following structure

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
      bool is_remote(std::size_t qpuId = 0);
      bool is_emulated(std::size_t qpuId = 0) const;
      std::string name() const;
 
      std::size_t get_current_qpu() const ;
      void set_current_qpu(const std::size_t device_id);

    };
  }

CUDA Quantum provides the following public functions to interact with the current
:code:`cudaq::quantum_platform`

.. code-block:: cpp

  namespace cudaq {
    quantum_platform &get_platform() ;
  }

