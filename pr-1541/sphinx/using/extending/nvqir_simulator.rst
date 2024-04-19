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
simulation-strategy-specific manner (e.g., state vector, tensor network, etc.). Moreover, 
subtypes are free to implement simulators that leverage classical accelerated computing. 

In this document, we'll detail this simulator interface and walk through how to extend it for new types of simulation. 

:code:`CircuitSimulator`
------------------------

The :code:`CircuitSimulator` type is defined in :code:`runtime/nvqir/CircuitSimulator.h`. It
exposes a public API to `libnvqir` that is immediately subclassed in the :code:`CircuitSimulatorBase` 
type. This type is templated on the floating point type used in the simulator's computations (e.g. :code:`double,float`).
This templated type handles a lot of the base functionality required for allocating and deallocated qubits, 
as well as measurement, sampling, and observation under a number of execution contexts. 
This is the type that downstream simulation developers should extend. 

The actual definition of the quantum state data structure, and its overall evolution are 
left as tasks for :code:`CircuitSimulatorBase` subclasses. Examples of simulation subtypes can be found 
in :code:`runtime/nvqir/qpp/QppCircuitSimulator.cpp` or :code:`runtime/nvqir/custatevec/CuStateVecCircuitSimulator.cpp`.
The :code:`QppCircuitSimulator` models the state vector using the `Q++ <https://github.com/softwareqinc/qpp>`_ library, which 
leverages the :code:`Eigen::Matrix` type and OpenMP threading for matrix-vector operations. 
The :code:`CuStateVecCircuitSimulator` type models the state vector on an NVIDIA GPU device 
by leveraging the cuQuantum library. 

The key methods that need to be overridden by subtypes of :code:`CircuitSimulatorBase` are as follows:

.. list-table:: Required Circuit Simulator Subtype Method Overrides

    * - Method Name 
      - Method Arguments
      - Method Description
    * - :code:`addQubitToState`
      - :code:`void` 
      - Add a qubit to the underlying state representation.
    * - :code:`addQubitsToState`
      - :code:`nQubits : std::size_t` 
      - Add the specified number of qubits to the underlying state representation.
    * - :code:`doResetQubit`
      - :code:`qubitIdx : std::size_t`
      - Reset the state of the qubit at the given index to :code:`|0>`
    * - :code:`resetQubitStateImpl`
      - :code:`void` 
      - Clear the entire state representation (reset to 0 qubits).
    * - :code:`applyGate`
      - :code:`task : GateApplicationTask`
      - Apply the specified gate described by the :code:`GateApplicationTask`. This type encodes the control and target qubit indices, optional rotational parameters, and the gate matrix data. 
    * - :code:`measureQubit`
      - :code:`qubitIdx : std::size_t -> bool` (returns bit result as bool)
      - Measure the qubit, produce a bit result, collapse the state.
    * - :code:`sample`
      - :code`qubitIdxs : std::vector<std::size_t>, shots : int`
      - Sample the current multi-qubit state on the provided qubit indices over a certain number of shots
    * - :code:`name`
      - :code:`void`
      - Return the name of this CircuitSimulator, must be the same as the name used in :code:`nvq++ -qpu NAME ...`

To extend a subtype class, you will need to create a new :code:`cpp` implementation file with the same name as your 
subtype class name. In this file, you will subclass the :code:`CircuitSimulatorBase<FloatType>` and implement the methods in 
the above table. Finally, the subclass must be registered with the NVQIR library so that it 
can be picked up and used when a user specifies :code:`nvq++ --target mySimulator ...` from the command line (or :code:`cudaq.set_target('mySimulator')` in Python.)
Type registration can be performed with a provided NVQIR macro, 

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

The library must be installed in :code:`$CUDA_QUANTUM_PATH/lib` and the configuration file 
must be installed to :code:`$CUDA_QUANTUM_PATH/platforms`.

Let's see this in action 
------------------------

CUDA Quantum provides some CMake utilities to make the creation of your new simulation library 
easier. Specifically, by using :code:`find_package(NVQIR)`, you'll get access to a :code:`nvqir_add_backend` function
that will automate much of the boilerplate for creating your library and configuration file.

Let's assume you want a simulation subtype named :code:`MySimulator`. You can create a folder or 
repository for this code called :code:`my-simulator` and add :code:`MySimulator.cpp` and 
:code:`CMakeLists.txt` files. Fill the CMake file with the following: 

.. code:: cmake 

    cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
    project(DemoCreateNVQIRBackend VERSION 1.0.0 LANGUAGES CXX)
    find_package(NVQIR REQUIRED)
    nvqir_add_backend(MySimulator MySimulator.cpp "")

and then fill out your :code:`MySimulator.cpp` file with your subtype implementation. For example, 

.. code:: cpp

    #include "CircuitSimulator.h"

    namespace {

      class MySimulator : public nvqir::CircuitSimulatorBase<double> {

      protected:
        /// @brief Grow the state vector by one qubit.
        void addQubitToState() override { ... }

        /// @brief Grow the state vector by `count` qubit.
        void addQubitsToState(std::size_t count) override { ... }

        /// @brief Reset the qubit state.
        void resetQubitStateImpl() override { ... }

        /// @brief Apply the given gate
        void applyGate(const GateApplicationTask &task) override { ... }

        /// @brief Reset a qubit to the |0> state.
        void doResetQubit(std::size_t qubitIdx) override { ... }

      public:
        MySimulator() = default;
        virtual ~MySimulator() = default;
        
        bool measureQubit(std::size_t qubitIdx) override { ... }

        cudaq::SampleResult sample(std::vector<std::size_t> &measuredBits,
                              int shots) override { ... }

        const std::string_view name() const override { return "MySimulator"; }
      
      };

    } // namespace

    /// Register this Simulator with NVQIR.
    NVQIR_REGISTER_SIMULATOR(MySimulator)

To build, install, and use this simulation backend, run the following from the top-level of :code:`my-simulator`:

.. code:: bash 

    export CUDA_QUANTUM_PATH=/path/to/cuda_quantum/install
    mkdir build && cd build 
    cmake .. -G Ninja -DNVQIR_DIR="$CUDA_QUANTUM_PATH/lib/cmake/nvqir"
    ninja install 

Then given any CUDA Quantum source file, you can compile and target your backend simulator with the following: 

.. code:: bash 

    nvq++ file.cpp --target MySimulator 
    ./a.out 
