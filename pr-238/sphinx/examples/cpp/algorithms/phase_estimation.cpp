// Compile and run with:
// ```
// nvq++ phase_estimation.cpp -o qpe.x && ./qpe.x
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <stdio.h>

#include <cmath>

// A pure device quantum kernel defined as a free function
// (cannot be called from host code).
__qpu__ void iqft(cudaq::qspan<> q) {
  int N = q.size();
  // Swap qubits
  for (int i = 0; i < N / 2; ++i) {
    swap(q[i], q[N - i - 1]);
  }

  for (int i = 0; i < N - 1; ++i) {
    h(q[i]);
    int j = i + 1;
    for (int y = i; y >= 0; --y) {
      double denom = (1UL << (j - y));
      const double theta = -M_PI / denom;
      r1<cudaq::ctrl>(theta, q[j], q[y]);
    }
  }

  h(q[N - 1]);
}

// CUDA Quantum kernel call operators can be templated on
// input CUDA Quantum kernel expressions. Here we define a general
// Phase Estimation algorithm that is generic on the eigenstate
// preparation and unitary evolution steps.
struct qpe {

  // Define the CUDA Quantum call expression to take user-specified eigenstate
  // and unitary evolution kernels, as well as the number of qubits in the
  // counting register and in the eigenstate register.
  template <typename StatePrep, typename Unitary>
  void operator()(const int nCountingQubits, StatePrep &&state_prep,
                  Unitary &&oracle) __qpu__ {
    // Allocate a register of qubits
    cudaq::qreg q(nCountingQubits + 1);

    // Extract sub-registers, one for the counting qubits
    // another for the eigenstate register
    auto counting_qubits = q.front(nCountingQubits);
    auto &state_register = q.back();

    // Prepare the eigenstate
    state_prep(state_register);

    // Put the counting register into uniform superposition
    h(counting_qubits);

    // Perform `ctrl-U^j`
    for (int i = 0; i < nCountingQubits; ++i) {
      for (int j = 0; j < (1 << i); ++j) {
        cudaq::control(oracle, counting_qubits[i], state_register);
      }
    }

    // Apply inverse quantum Fourier transform
    iqft(counting_qubits);

    // Measure to gather sampling statistics
    mz(counting_qubits);

    return;
  }
};

struct r1PiGate {
  void operator()(cudaq::qubit &q) __qpu__ { r1(1., q); }
};

int main() {

  for (auto nQubits : std::vector<int>{2, 4, 6, 8}) {
    auto counts = cudaq::sample(
        qpe{}, nQubits, [](cudaq::qubit &q) __qpu__ { x(q); }, r1PiGate{});
    auto mostProbable = counts.most_probable();
    double theta = cudaq::to_integer(mostProbable) / (double)(1UL << nQubits);
    auto piEstimate = 1. / (2 * theta);
    printf("Pi Estimate(nQubits == %d) = %lf \n", nQubits, piEstimate);
  }
}
