
Example Programs
****************
Hello World - Simple Bell State
-------------------------------

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

GHZ State Preparation and Sampling
----------------------------------

.. code-block:: cpp 
 
  #include <cudaq.h>

  struct ghz {
    void operator()(const int n_qubits) __qpu__ {
      cudaq::qvector q(n_qubits);
      h(q[0]);
      for (int i = 0; i < n_qubits - 1; ++i) 
        // note use of ctrl modifier
        x<cudaq::ctrl>(q[i], q[i+1]); 
      
      mz(q);
    }
  };

  int main() {
    // Sample the state produced by the ghz kernel
    auto counts = cudaq::sample(ghz{}, 10);
    for (auto [bits, count] : counts) {
      printf("Observed %s %lu times.\n", bits.c_str(), count);
    }
    return 0;
  }

Quantum Phase Estimation
------------------------

.. literalinclude:: ../../examples/cpp/algorithms/phase_estimation.cpp
    :language: cpp

Deuteron Binding Energy Parameter Sweep
---------------------------------------

.. code-block:: cpp 

  #include <cudaq.h>
  #include <cudaq/algorithm.h>

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


Grover's Algorithm
------------------

.. literalinclude:: ../../examples/cpp/algorithms/grover.cpp
    :language: cpp

Iterative Phase Estimation
--------------------------

.. literalinclude:: ../../examples/cpp/other/iterative_qpe.cpp
    :language: cpp
    :start-after: [Begin Documentation]
    :end-before: [End Documentation]