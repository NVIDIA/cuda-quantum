
Example Programs
****************
Hello World - Simple Bell State
-------------------------------

.. tab:: C++ 

  .. code-block:: cpp

      #include <cudaq.h>

      struct bell {
        int operator()(int num_iters) __qpu__ {
          cudaq::qarray<2> q;
          int nCorrect = 0;
          for (int i = 0; i < num_iters; i++) {
            h(q[0]);
            x<cudaq::ctrl>(q[0], q[1]);
            auto results = mz(q);
            if (results[0] == results[1]) 
              nCorrect++;
            
            reset(q[0]);
            reset(q[1]);
          }
          return nCorrect;
        }
      };

      int main() { printf("N Correct = %d\n", bell{}(100)); }

.. tab:: Python 

  .. code-block:: python 

    import cudaq 

    @cudaq.kernel()
    def bell(num_iters : int) -> int:
        q = cudaq.qvector(2)
        nCorrect = 0
        for i in range(num_iters):
            h(q[0])
            x.ctrl(q[0], q[1])
            results = mz(q)
            if results[0] == results[1]:
               nCorrect = nCorrect + 1
            
            reset(q)
        return nCorrect 

    counts = bell(100)
    print(f'N Correct = {counts}')
    assert counts == 100

GHZ State Preparation and Sampling
----------------------------------

.. tab:: C++ 

  .. code-block:: cpp 
  
      #include <cudaq.h>

      __qpu__ ghz(const int n_qubits) {
        cudaq::qvector q(n_qubits);
        h(q[0]);
        for (int i = 0; i < n_qubits - 1; ++i) 
          // note use of ctrl modifier
          x<cudaq::ctrl>(q[i], q[i+1]); 
        
        mz(q);
      }

      int main() {
        // Sample the state produced by the ghz kernel
        auto counts = cudaq::sample(ghz, 10);
        for (auto [bits, count] : counts) {
          printf("Observed %s %lu times.\n", bits.c_str(), count);
        }
        return 0;
      }

.. tab:: Python 

  .. code-block:: python 

    import cudaq 
    
    @cudaq.kernel
    def ghz(numQubits:int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits - 1)):
            x.ctrl(qubit, qubits[i + 1])
    
    counts = cudaq.sample(ghz, 10)
    for bits, count : counts:
        print('Observed {} {} times.'.format(bits, count))

Quantum Phase Estimation
------------------------

.. tab:: C++ 

  .. literalinclude:: ../../examples/cpp/algorithms/phase_estimation.cpp
      :language: cpp

.. tab:: Python 

  .. code-block:: python 

    import cudaq, numpy as np

    # Compute phase for U |psi> = exp(-2 pi phase) |psi> 
    # This example will consider U = T, and |psi> = |1>
    # Define a Inverse Quantum Fourier Transform kernel
    @cudaq.kernel
    def iqft(qubits: cudaq.qview):
        N = qubits.size()
        for i in range(N // 2):
            swap(qubits[i], qubits[N - i - 1])

        for i in range(N - 1):
            h(qubits[i])
            j = i + 1
            for y in range(i, -1, -1):
                r1.ctrl(-np.pi / 2**(j - y), qubits[j], qubits[y])

        h(qubits[N - 1])


    # Define the U kernel
    @cudaq.kernel
    def tGate(qubit: cudaq.qubit):
        t(qubit)


    # Define the state preparation |psi> kernel
    @cudaq.kernel
    def xGate(qubit: cudaq.qubit):
        x(qubit)

    # General Phase Estimation kernel for single qubit 
    # eigen states. 
    @cudaq.kernel
    def qpe(nC: int, nQ: int, statePrep: Callable[[cudaq.qubit], None],
            oracle: Callable[[cudaq.qubit], None]):
        q = cudaq.qvector(nC + nQ)
        countingQubits = q.front(nC)
        stateRegister = q.back()
        statePrep(stateRegister)
        h(countingQubits)
        for i in range(nC):
            for j in range(2**i):
                cudaq.control(oracle, [countingQubits[i]], stateRegister)
        iqft(countingQubits)
        mz(countingQubits)

    # Sample the state to get the phase. 
    counts = cudaq.sample(qpe, 3, 1, xGate, tGate)
    assert len(counts) == 1
    assert '100' in counts


Deuteron Binding Energy Parameter Sweep
---------------------------------------

.. tab:: C++ 

  .. code-block:: cpp 

      #include <cudaq.h>

      struct deuteron_n2_ansatz {
        void operator()(double theta) __qpu__ {
          cudaq::qarray<2> q;
          x(q[0]);
          ry(theta, q[1]);
          x<cudaq::ctrl>(q[1], q[0]);
        }
      };

      int main() {
        using namespace cudaq::spin;
        cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                    .21829 * z(0) - 6.125 * z(1);
    
        // Perform parameter sweep for deuteron N=2 Hamiltonian
        const auto param_space = cudaq::linspace(-M_PI, M_PI, 25);
        for (const auto& param : param_space) {
          // KERNEL::observe(...) <==> 
          // E(params...) = <psi(params...) | H | psi(params...)>
          double energy_at_param = cudaq::observe(deuteron_n2_ansatz{}, h, param);
          printf("<H>(%lf) = %lf\n", param, energy_at_param);
        }
      }

.. tab:: Python 

  .. code-block:: python 

    import cudaq 
    from cudaq import spin 

    @cudaq.kernel
    def ansatz(angle:float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(angle, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Perform parameter sweep for deuteron N=2 Hamiltonian
    for angle in np.linspace(-np.pi, np.pi, 25):
         # KERNEL::observe(...) <==> 
         # E(params...) = <psi(params...) | H | psi(params...)>
         energyAtParam = cudaq.observe(ansatz, hamiltonian, .59)
         print('<H>({}) = {}'.format(angle, energyAtParam))


Grover's Algorithm
------------------

.. tab:: C++ 

  .. literalinclude:: ../../examples/cpp/algorithms/grover.cpp
      :language: cpp

.. tab:: Python 

  .. code-block:: python 

    @cudaq.kernel 
    def reflect(qubits: cudaq.qview):
        ctrls = qubits.front(qubits.size() - 1)
        last = qubits.back()
        cudaq.compute_action(lambda: (h(qubits), x(qubits)),
                              lambda: z.ctrl(ctrls, last))

    @cudaq.kernel
    def oracle(q: cudaq.qview):
        z.ctrl(q[0], q[2])
        z.ctrl(q[1], q[2])

    @cudaq.kernel
    def grover(N: int, M: int, oracle: Callable[[cudaq.qview], None]):
        q = cudaq.qvector(N)
        h(q)
        for i in range(M):
            oracle(q)
            reflect(q)
        mz(q)

    counts = cudaq.sample(grover, 3, 1, oracle)
    assert len(counts) == 2
    assert '101' in counts
    assert '011' in counts


Iterative Phase Estimation
--------------------------

.. literalinclude:: ../../examples/cpp/other/iterative_qpe.cpp
    :language: cpp
    :start-after: [Begin Documentation]
    :end-before: [End Documentation]
