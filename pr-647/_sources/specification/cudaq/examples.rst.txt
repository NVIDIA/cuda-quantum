
Example Programs
****************
Hello World - Simple Bell State
-------------------------------

.. code-block:: cpp

  #include <cudaq.h>

  struct bell {
    int operator()(int num_iters) __qpu__ {
      cudaq::qreg q(2);
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
      cudaq::qreg q(n_qubits);
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

.. code-block:: cpp 

  #include <cudaq.h>
  #include <cmath>

  __qpu__ void iqft (cudaq::qspan<> q) {
    int N = q.size();
    // Swap qubits
    for (int i = 0; i < N / 2; ++i) {
      swap(q[i], q[N - i - 1]);
    }

    for (int i = 0; i < N - 1; ++i) {
      h(q[i]);
      int j = i + 1;
      for (int y = i; y >= 0; --y) {
        const double theta = -M_PI / std::pow(2.0, j - y);
        r1<cudaq::ctrl>(theta, q[j], q[y]);
      }
    }

    h(q[N - 1]);
  }

  struct PhaseEstimation {
    double operator()(const int n_counting_qubits, const int n_state_qubits,
                     auto&& statePrep, auto&& unitary) __qpu__ {
      // Allocate a register of qubits
      cudaq::qreg q(n_counting_qubits + n_state_qubits);

      // Extract sub-registers, one for the counting qubits
      // another for the eigen state register
      auto counting_qubits = q.front(n_counting_qubits);
      auto state_register = q.back(n_state_qubits);

      // Prepare the eigenstate 
      statePrep(state_register);

      // Put the counting register into uniform superposition
      h(counting_qubits);

      // Perform ctrl-U^j
      for (int i = 0; i < n_counting_qubits; ++i) {
        for (int j = 0; j < 1UL << i; ++j) {
          cudaq::control(unitary, counting_qubits[i], state_register);
        }
      }

      // Apply inverse quantum fourier transform
      iqft(counting_qubits);

      // Measure and compute the phase...
      auto bits = mz(counting_qubits);
      return cudaq::to_integer(bits) / std::pow(2, n_counting_qubits);
    }
  };

  int main() {
    auto statePrep = [](cudaq::qspan<> q) __qpu__ { x(q); };
    auto unitary = [](cudaq::qspan<> q) __qpu__ { t(q); };
    double phase = PhaseEstimation{}(3, 1, statePrep, unitary);
    printf("Phase = %lf\n", phase);
  }

Deuteron Binding Energy Parameter Sweep
---------------------------------------

.. code-block:: cpp 

  #include <cudaq.h>
  #include <cudaq/algorithm.h>

  struct deuteron_n2_ansatz {
    void operator()(double theta) __qpu__ {
      cudaq::qreg q(2);
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

.. code-block:: cpp 

  #include <cudaq.h>

  __qpu__ void reflect_about_uniform(cudaq::qspan<> q) {
    auto ctrl_qubits = q.front(q.size() - 1);
    auto& last_qubit = q.back();
    // Compute (U) Action (V) produces
    // U V U::Adjoint
    cudaq::compute_action([&]() {
                           h(q);
                           x(q);
                         },
                         [&]() { 
                           z<cudaq::ctrl>(ctrl_qubits, last_qubit); 
                         }
                        );
  }

  struct run_grover {
    template <typename CallableKernel>
    auto operator()(const int n_qubits, const int n_iterations,
                          CallableKernel&& oracle) __qpu__ {
      cudaq::qreg q(n_qubits);
      h(q);
      for (int i = 0; i < n_iterations; i++) {
        oracle(q);
        reflect_about_uniform(q);
      }
      mz(q);
    }
  };

  struct oracle {
    void operator()(cudaq::qspan<> q) __qpu__ {
      cz(q[0], q[2]);
      cz(q[1], q[2]);
    }
  };

  int main() {
    auto counts = cudaq::sample(run_grover{}, 3, 1, oracle{});
    counts.dump();
    return 0;
  }

Iterative Phase Estimation
--------------------------

.. code-block:: cpp

   #include <cudaq.h>

  struct iqpe {
    void operator()() __qpu__ {
      cudaq::qreg<2> q;
      h(q[0]);
      x(q[1]);
      for (int i = 0; i < 8; i++)
        r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

      h(q[0]);
      auto cr0 = mz(q[0]);
      reset(q[0]);

      h(q[0]);
      for (int i = 0; i < 4; i++)
        r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

      if (cr0)
        rz(-M_PI / 2., q[0]);

      h(q[0]);
      auto cr1 = mz(q[0]);
      reset(q[0]);

      h(q[0]);
      for (int i = 0; i < 2; i++)
        r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

      if (cr0)
        rz(-M_PI / 4., q[0]);
  
      if (cr1)
        rz(-M_PI / 2., q[0]);

      h(q[0]);
      auto cr2 = mz(q[0]);
      reset(q[0]);
      h(q[0]);
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

      if (cr0)
        rz(-M_PI / 8., q[0]);

      if (cr1)
        rz(-M_PI_4, q[0]);

      if (cr2)
        rz(-M_PI_2, q[0]);

      h(q[0]);
      mz(q[0]);
    }
  };

  int main() {
    auto counts = cudaq::sample(/*shots*/ 100, iqpe{});
    counts.dump();
    return 0;
  }
