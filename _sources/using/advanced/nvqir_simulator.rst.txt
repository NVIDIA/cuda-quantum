Extending CUDA Quantum with a new Simulator
*******************************************

Backend circuit simulation in CUDA Quantum is enabled via the 
NVQIR library (:code:`libnvqir`). CUDA Quantum code is ultimately lowered 
to the LLVM IR in a manner that is adherent to the `QIR specification <https://qir-alliance.org>`_.
NVQIR provides function implementations for the various declared functions 
in the specification, which in turn delegate to an extensible simulation 
architecture. 

The primary extension point for NVQIR is the :code:`CircuitSimulator` class. This class
exposes an API that enables qubit allocation and deallocation, quantum operation 
invocation, and measurement and sampling. Subtypes of this class are free to 
override these methods to affect simulation of the quantum code in any 
simulation strategy specific manner (e.g. state vector, tensor network, etc.). Moreover, 
subtypes are free to implement simulators that leverage classical accelerated computing. 

In this document, we'll detail this simulator interface and walk through how one 
might extend it for new types of simulation. 

CircuitSimulator
----------------

The :code:`CircuitSimulator` type is defined in :code:`runtime/nvqir/CircuitSimulator.h`. 
It handles a lot of the base functionality required for allocating and deallocated qubits, 
as well as measurement, sampling, and observation under a number of execution contexts. 

Actual definition of the quantum state data structure, and its overall evolution are 
left as tasks for subclasses. Examples of simulation subtypes can be found 
in :code:`runtime/nvqir/qpp/QppCircuitSimulator.cpp` or :code:`runtime/nvqir/custatevec/CuStateVecCircuitSimulator.cpp`.
The :code:`QppCircuitSimulator` models the state vector using the `Q++ <https://github.com/softwareqinc/qpp>`_ library, which 
boils down to an :code:`Eigen::Matrix` type and leverages OpenMP threading for matrix-vector operations. 
The :code:`CuStateVecCircuitSimulator` type models the state vector on an NVIDIA GPU device 
by leveraging the cuQuantum library. 

The key methods that need to be overridden by subtypes of :code:`CircuitSimulator` are as follows:

.. list-table:: Required Circuit Simulator Subtype Method Overrides

    * - Method Name 
      - Method Arguments
      - Method Description
    * - :code:`addQubitToState`
      - :code:`void` 
      - Add a qubit to the underlying state representation.
    * - :code:`resetQubit`
      - :code:`qubitIdx : std::size_t`
      - Reset the state of the qubit at the given index to :code:`|0>`
    * - :code:`resetQubitStateImpl`
      - :code:`void` 
      - Clear the entire state representation (reset to 0 qubits).
    * - Quantum Operation Methods
      - varies per overload - target qubit index, control qubit index (indices), rotation parameters
      - Apply the unitary described by the operation with given method name. (e.g. h(qubitIdx), apply hadamard on qubit with index qubitIdx)  
    * - :code:`measureQubit`
      - :code:`qubitIdx : std::size_t -> bool` (returns bit result as bool)
      - Measure the qubit, produce a bit result, collapse the state.
    * - :code:`sample`
      - qubitIdxs : std::vector<std::size_t>, shots : int 
      - Sample the current multi-qubit state on the provided qubit indices over a certain number of shots
    * - :code:`name`
      - :code:`void`
      - Return the name of this CircuitSimulator, must be the same as the name used in :code:`nvq++ -qpu NAME ...`

The strategy for extending this class is to create a new :code:`cpp` implementation file with the same name as your 
subtype class name. In this file, you will subclass the :code:`CircuitSimulator` and implement the methods in 
the above table. Finally, the subclass must be registered with the NVQIR library so that it 
can be picked up and used when a user specifies :code:`nvq++ -qpu mySimulator ...` from the command line (or :code:`cudaq.set_qpu('mySimulator')` in Python.)
Type registration can be performed with a provided NVQIR macro 

.. code:: cpp 

    NVQIR_REGISTER_SIMULATOR(MySimulatorClassName, mySimulator)

where :code:`MySimulatorClassName` is the name of your subtype, and :code:`mySimulator` is the 
same name as what :code:`MySimulatorClassName::name()` returns, and what you desire the 
:code:`-qpu NAME` name to be. 

A further requirement is that the code be compiled into its own standalone shared library 
with name :code:`libnvqir-NAME.{so,dylib}`, where NAME is the 
same name as what :code:`MySimulatorClassName::name()` returns, and what you desire the 
:code:`-qpu NAME` name to be. You will also need to create a :code:`NAME.config` file that 
contains the following contents 

.. code:: bash 

    NVQIR_SIMULATION_BACKEND="NAME"

The library must be installed in :code:`CUDAQ_INSTALL/lib` and the configuration file 
must be installed to :code:`CUDAQ_INSTALL/platforms`.

Let's see this in action 
------------------------

CUDA Quantum provides some CMake utilities to make the creation of your new simulation library 
easier. Specifically, but using :code:`find_package(NVQIR)`, you'll get access to a :code:`nvqir_add_backend` function
that will automate a lot of the boilerplate for creating your library and config file.

Let's assume you want a simulation subtype named :code:`MySimulator`. You can create a folder or 
repository for this code called :code:`my-simulator` and add :code:`MySimulator.cpp` and 
:code:`CMakeLists.txt` files. Fill the CMake file with the following 

.. code:: cmake 

    cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
    project(DemoCreateNVQIRBackend VERSION 1.0.0 LANGUAGES CXX)
    find_package(NVQIR REQUIRED)
    nvqir_add_backend(MySimulator MySimulator.cpp)

and then fill out your :code:`MySimulator.cpp` file with your subtype implementation, something like 

.. code:: cpp

    #include "CircuitSimulator.h"

    namespace {

      class MySimulator : public nvqir::CircuitSimulator {

      protected:
        /// @brief Grow the state vector by one qubit.
        void addQubitToState() override { ... }

        /// @brief Reset the qubit state.
        void resetQubitStateImpl() override { ... }

      public:
        MySimulator() = default;
        virtual ~MySimulator() = default;

      /// The one-qubit overrides
      #define ONE_QUBIT_METHOD_OVERRIDE(NAME)                                        \
        using CircuitSimulator::NAME;                                                \
        virtual void NAME(std::vector<std::size_t> &controls, std::size_t &qubitIdx) \
            override { ... }

      ONE_QUBIT_METHOD_OVERRIDE(x)
      ONE_QUBIT_METHOD_OVERRIDE(y)
      ONE_QUBIT_METHOD_OVERRIDE(z)
      ONE_QUBIT_METHOD_OVERRIDE(h)
      ONE_QUBIT_METHOD_OVERRIDE(s)
      ONE_QUBIT_METHOD_OVERRIDE(t)
      ONE_QUBIT_METHOD_OVERRIDE(sdg)
      ONE_QUBIT_METHOD_OVERRIDE(tdg)

      /// The one-qubit parameterized overrides
      #define ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(NAME)                         \
      using CircuitSimulator::NAME;                                             \
      virtual void NAME(double &angle, std::vector<std::size_t> &controls,      \
                       std::size_t &qubitIdx) override { ... }

      ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(rx)
      ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(ry)
      ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(rz)
      ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(r1)
      ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(u1)

      /// @brief U2 operation
      using CircuitSimulator::u2;
      void u2(double &phi, double &lambda, std::vector<std::size_t> &controls,
          std::size_t &qubitIdx) override { ... }

      /// @brief U3 operation
      using CircuitSimulator::u3;
      void u3(double &theta, double &phi, double &lambda,
          std::vector<std::size_t> &controls, std::size_t &qubitIdx) override { ... }

      /// @brief Swap operation
      using CircuitSimulator::swap;
      void swap(std::vector<std::size_t> &ctrlBits, std::size_t &srcIdx,
            std::size_t &tgtIdx) override { ... }

      bool measureQubit(std::size_t qubitIdx) override { ... }

      void resetQubit(std::size_t &qubitIdx) override { ... }

      cudaq::SampleResult sample(std::vector<std::size_t> &measuredBits,
                              int shots) override { ... }

      const std::string_view name() const override { return "MySimulator"; }
    };

    } // namespace

    /// Register this Simulator with NVQIR.
    NVQIR_REGISTER_SIMULATOR(MySimulator)

To build, install, and use this simulation backend, run the following from the top-level of :code:`my-simulator`

.. code:: bash 

    mkdir build && cd build 
    cmake .. -G Ninja -DNVQIR_DIR=$CUDAQ_INSTALL/lib/cmake/nvqir 
    ninja install 

Then given any CUDA Quantum source file, you can compile and target your backend simulator with 

.. code:: bash 

    nvq++ file.cpp --qpu MySimulator 
    ./a.out 
