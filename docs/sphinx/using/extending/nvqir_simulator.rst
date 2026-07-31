.. _nvqir_simulator:
:spellcheck-disable:
================================
Extending NVQIR with Simulators
================================

Backend circuit simulation in CUDA-Q is enabled via the 
NVQIR library (:code:`libnvqir`). CUDA-Q code is ultimately lowered 
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

In this document, we'll detail this simulator interface and walk through how to extend it. 

The :code:`CircuitSimulator` type is defined in :code:`runtime/nvqir/CircuitSimulator.h`. It
exposes a public API to `libnvqir` that is immediately subclassed in the :code:`CircuitSimulatorBase` 
type. This type is templated on the floating point type used in the simulator's computations (e.g. :code:`double,float`).
This templated type handles a lot of the base functionality required for allocating and deallocating qubits, 
as well as measurement, sampling, and observation under a number of execution contexts. 
This is the type that downstream simulation developers should extend. 

The actual definition of the quantum state data structure, and its overall evolution are 
left as tasks for :code:`CircuitSimulatorBase` subclasses. Examples of simulation subtypes can be found 
in :code:`runtime/nvqir/qpp/QppCircuitSimulator.cpp` (a full-featured CPU state-vector simulator built on 
`Q++ <https://github.com/softwareqinc/qpp>`_) or :code:`runtime/nvqir/custatevec/CuStateVecCircuitSimulator.cpp` 
(a GPU state-vector simulator built on cuQuantum). :code:`runtime/nvqir/resourcecounter/ResourceCounter.h` is 
also a good, much smaller reference: it implements every required method with only a few lines each.

The methods that must be overridden by every subtype of :code:`CircuitSimulatorBase` are as follows.
Everything else declared in :code:`CircuitSimulator` already has a default implementation in 
:code:`CircuitSimulatorBase` and does not need to be touched by a typical subclass.

.. list-table:: Required Circuit Simulator Subtype Method Overrides

    * - Method Name 
      - Method Signature
      - Method Description
    * - :code:`addQubitToState`
      - :code:`void addQubitToState()`
      - Grow the underlying state representation by one qubit, initialized to :code:`|0>`.
    * - :code:`deallocateStateImpl`
      - :code:`void deallocateStateImpl()`
      - Clear the entire state representation (invoked when all qubits have been deallocated).
    * - :code:`applyGate`
      - :code:`void applyGate(const GateApplicationTask &task)`
      - Apply the specified gate described by the :code:`GateApplicationTask`. This type encodes the
        control and target qubit indices, optional rotational parameters, and the dense gate matrix data.
    * - :code:`measureQubit`
      - :code:`bool measureQubit(const std::size_t qubitIdx)`
      - Measure the qubit, produce a bit result, and collapse the state.
    * - :code:`setToZeroState`
      - :code:`void setToZeroState()`
      - Reset the whole state back to :code:`|0...0>`.
    * - :code:`resetQubit`
      - :code:`void resetQubit(const std::size_t qubitIdx)`
      - Reset the state of the qubit at the given index to :code:`|0>`, in place.
    * - :code:`sample`
      - :code:`cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits, const int shots, bool includeSequentialData = true)`
      - Sample the current multi-qubit state on the provided qubit indices over a certain number of shots.
    * - :code:`name`
      - :code:`std::string name() const`
      - Return the name of this :code:`CircuitSimulator`. Must be the same as the name used in :code:`nvq++ --target NAME ...`.
    * - :code:`createStateFromData`
      - :code:`std::unique_ptr<cudaq::SimulationState> createStateFromData(const cudaq::state_data &)`
      - Construct a simulator-specific :code:`cudaq::SimulationState` from user-supplied data. If your
        simulator doesn't need to support this, it is enough to throw an exception here.
    * - :code:`clone`
      - :code:`nvqir::CircuitSimulator *clone()`
      - Return a simulator instance to use on a given execution thread. This is almost always generated
        for you by the :code:`NVQIR_SIMULATOR_CLONE_IMPL` macro (see below) rather than written by hand.

A couple of methods are declared with a default implementation but can be overridden if it's useful for
your simulation strategy, notably :code:`addQubitsToState` (batch qubit allocation; defaults to calling
:code:`addQubitToState` in a loop), :code:`canHandleObserve` (whether the simulator can compute
:code:`<psi|H|psi>` directly instead of via measurement sampling; defaults to :code:`false`), and
:code:`observe`, :code:`allocateQubit(s)`, :code:`deallocateQubits`, and the :code:`finalizeExecutionContext`
overloads, all of which already have working implementations in :code:`CircuitSimulatorBase` built on
top of the methods above.

To extend a subtype class, you will need to create a new :code:`cpp` implementation file with the same name as your 
subtype class name. In this file, you will subclass :code:`nvqir::CircuitSimulatorBase<FloatType>` and implement the 
methods in the above table. Finally, the subclass must be registered with the NVQIR library so that it 
can be picked up and used when a user specifies :code:`nvq++ --target MySimulator ...` from the command line 
(or :code:`cudaq.set_target('MySimulator')` in Python.) Type registration can be performed with a provided NVQIR macro, 

.. code:: cpp 

    NVQIR_REGISTER_SIMULATOR(MySimulatorClassName, PrintedName)

where :code:`MySimulatorClassName` is the name of your subtype, and :code:`PrintedName` is the 
same name as what :code:`MySimulatorClassName::name()` returns, and what you desire the 
:code:`--target NAME` name to be. If you use the :code:`NVQIR_SIMULATOR_CLONE_IMPL` convenience macro 
to implement :code:`clone()` (recommended, see the example below), pass it your class name the same way:

.. code:: cpp

    NVQIR_SIMULATOR_CLONE_IMPL(MySimulatorClassName)

A further requirement is that the code be compiled into its own standalone shared library 
with the name :code:`libnvqir-NAME.{so,dylib}`, where :code:`NAME` is the 
same name as what :code:`MySimulatorClassName::name()` returns, and what you desire the 
:code:`--target NAME` name to be. NVQIR also needs a small target configuration file describing the
backend (its name and whether it requires a GPU); as of CUDA-Q 0.12, this file is generated for you 
automatically by the CMake helper below, so you do not need to author it by hand.

The library and its generated target configuration file must be installed to 
:code:`$CUDA_QUANTUM_PATH/lib` and :code:`$CUDA_QUANTUM_PATH/targets` respectively; the CMake helper 
below takes care of this for you as well.

Toolchain requirements
-----------------------

:code:`CircuitSimulator.h` uses C++20 language features throughout (concepts, :code:`requires`-clauses,
:code:`std::span`), so your simulator must be compiled as C++20. Linking against the :code:`nvqir::nvqir`
CMake target does **not** turn this on for you automatically, so set it explicitly in your
:code:`CMakeLists.txt`:

.. code:: cmake

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

There is a second, easy-to-miss requirement: the CUDA-Q binaries you are linking against were built with
Clang and `libc++ <https://libcxx.llvm.org>`_ (not the GNU standard library, `libstdc++`). If your build
picks up your system's default compiler (commonly :code:`g++` with `libstdc++` on Linux), configuration and
compilation will typically succeed, but installing and running your plugin will fail with a
:code:`symbol lookup error`, because `libc++` and `libstdc++` mangle C++ symbol names differently -- the
linker can locate the CUDA-Q libraries, but not the specific symbols your object file expects from them.

To avoid this, point CMake explicitly at the Clang and :code:`libc++` that ship alongside your CUDA-Q
installation:

.. code:: bash

    cmake .. -G Ninja \
      -DNVQIR_DIR="$CUDA_QUANTUM_PATH/lib/cmake/nvqir" \
      -DCMAKE_CXX_COMPILER="$CUDA_QUANTUM_PATH/lib/llvm/bin/clang++" \
      -DCMAKE_CXX_FLAGS="-stdlib=libc++"

If you instead see compile-time errors mentioning :code:`concept`, :code:`requires`, or :code:`std::span`
not being recognized, that is the C++20 setting above being missing rather than the toolchain -- check
that first.

Let's see this in action
========================

Let's assume you want a simulation subtype named :code:`MySimulator`. You can create a new 
repository for this code called :code:`my-simulator` and add :code:`MySimulator.cpp` and 
:code:`CMakeLists.txt` files. Fill the CMake file with the following: 

.. literalinclude:: ../../snippets/cpp/using/extending/nvqir_simulator/CMakeLists.txt
    :language: cmake
    :start-after: [Begin Documentation]
    :end-before: [End Documentation]

The second argument to :code:`nvqir_add_backend` is :code:`GPURequirements`: pass :code:`true` if your
backend requires an NVIDIA GPU to run, or :code:`false` if it runs on the CPU. Any further arguments are
the source files that make up your backend's shared library.

Then fill out your :code:`MySimulator.cpp` file with your subtype implementation. The example below is a
small, fully-working (if unoptimized) dense state-vector simulator that implements every required
method from the table above:

.. literalinclude:: ../../snippets/cpp/using/extending/nvqir_simulator/MySimulator.cpp
    :language: cpp
    :start-after: [Begin Documentation]
    :end-before: [End Documentation]


Understanding the Simulator Implementation
==========================================

If you are new to writing quantum simulators, the underlying mathematics in `MySimulator.cpp` might look complex. Here is a didactic, step-by-step breakdown of how the core mechanisms actually work.

1. **The State Vector Architecture**  
   Quantum states are represented in binary. The ``state`` variable is a dense array of complex numbers, where the array index directly maps to the physical state. For example, a 2‑qubit system has :math:`2^2 = 4` elements: index ``0`` (:math:`00_2`) is :math:`|00\rangle`, index ``1`` (:math:`01_2`) is :math:`|01\rangle`, and so on. The value at that index is the probability amplitude.

2. **Growing the State** – ``addQubitToState()``  
   When a new qubit is introduced, the number of possible states doubles. The code creates a new array twice the size of the old one, copies the existing amplitudes into the first half, and leaves the second half as zeros. This effectively initializes the new qubit in the :math:`|0\rangle` state while perfectly preserving the existing entanglement of the previous qubits.

3. **Applying a Quantum Gate** – ``applyGate()``  
   This is where the heavy lifting happens. A gate applies a matrix transformation to specific qubits.

   - **Masks:** The code generates ``controlMask`` and ``targetMask``. These act like stencils to easily isolate which bits of an array index correspond to the control and target qubits.
   - **Checking Controls:** It loops through every possible state index. By performing a `bitwise` AND (``idx & controlMask``), it checks if all control qubits are ``1``. If not, the gate ignores this state.
   - **Finding the Base Subspace:** Quantum gates mix the amplitudes of states that differ *only* in their target qubits. By applying an inverted target mask (``~targetMask``), the code zeros out the target bits to find the "base" index of this state grouping.
   - **Linear Algebra:** It gathers the relevant amplitudes into a small vector (``amplitudes``), multiplies them by the gate's dense matrix (``task.matrix``), and distributes the newly calculated values back into ``newState``.
   - **The Visited Array:** Because a :math:`4 \times 4` gate mixes 4 amplitudes together, the outer loop will naturally encounter these same 4 indices as it counts up. The ``visited`` array ensures we only perform the matrix multiplication once per subspace.

4. **Simulating Measurement** – ``measureQubit()``  
   Measuring a qubit collapses the superposition into a definite state.

   - **Calculate Probability:** It sums the squared magnitudes (``std::norm``) of all amplitudes where the target qubit is ``1``. This is the probability of measuring a ``1``.
   - **The Dice Roll:** It generates a random number between 0.0 and 1.0. If the random number is less than the calculated probability, the measurement result is ``1``; otherwise, it's ``0``.
   - **Collapse and Normalize:** Any state that contradicts the measurement result is overwritten with zero (destroyed). Because half the amplitudes were deleted, the total probability of the system is no longer 1. The code calculates a scaling factor and scales up the surviving amplitudes to `"renormalize"` the state vector.

5. **Sampling the State** – ``sample()``  
   Unlike measurement, sampling does *not* collapse the underlying quantum state. It works like Monte Carlo simulation: for every "shot", it rolls a random number and calculates the `Cumulative Distribution Function (CDF)` on the fly. It sweeps through the state vector, adding up probabilities until the sum exceeds the random number. The basis state it lands on is the observed result for that shot.


.. code:: bash 

    mkdir build && cd build 
    cmake .. -G Ninja \
      -DNVQIR_DIR="$CUDA_QUANTUM_PATH/lib/cmake/nvqir" \
      -DCMAKE_CXX_COMPILER="$CUDA_QUANTUM_PATH/lib/llvm/bin/clang++" \
      -DCMAKE_CXX_FLAGS="-stdlib=libc++"
    ninja install 

Then given any CUDA-Q source file, you can compile and target your backend simulator with the following: 

.. code:: bash 

    nvq++ file.cpp --target MySimulator 
    ./a.out 

Testing your simulator
======================

Before trusting a new simulator, run a small circuit whose output you can check by hand. A Bell state is
a good choice: it entangles two qubits so that only :code:`00` and :code:`11` should ever be observed,
each roughly half the time, and nothing else.

.. code:: cpp

    #include <cudaq.h>

    struct bell {
      void operator()() __qpu__ {
        cudaq::qvector q(2);
        h(q[0]);
        x<cudaq::ctrl>(q[0], q[1]);
        mz(q);
      }
    };

    int main() {
      auto counts = cudaq::sample(bell{});
      counts.dump();
    }

.. code:: bash

    nvq++ bell.cpp --target MySimulator -o bell
    ./bell
    # { 00:~500 11:~500 }

If you instead see a uniform split across all four outcomes, or a single deterministic outcome every
time, that is a strong signal the bug is in gate application or measurement, not in your build -- add
some `fprintf` diagnostics to :code:`applyGate`, :code:`measureQubit`, and :code:`sample` to print the
state vector at each step, and compare it against a hand calculation of what the circuit should produce.
:spellcheck-enable: