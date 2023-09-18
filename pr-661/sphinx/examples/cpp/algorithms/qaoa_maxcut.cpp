// Compile and run with:
// ```
// nvq++ qaoa_maxcut.cpp -o qaoa.x && ./qaoa.x
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>
#include <cudaq/spin_op.h>

// Here we build up a CUDA Quantum kernel for QAOA with p layers, with each
// layer containing the alternating set of unitaries corresponding to the
// problem and the mixer Hamiltonians. The algorithm leverages the CUDA Quantum
// VQE support to compute the Max-Cut of a rectangular graph illustrated below.
//
//        v0  0---------------------0 v1
//            |                     |
//            |                     |
//            |                     |
//            |                     |
//        v3  0---------------------0 v2
// The Max-Cut for this problem is 0101 or 1010.

struct ansatz {
  void operator()(std::vector<double> theta, const int n_qubits,
                  const int n_layers) __qpu__ {
    cudaq::qreg q(n_qubits);

    // Prepare the initial state by superposition
    h(q);

    // Loop over all the layers
    for (int i = 0; i < n_layers; ++i) {
      // Problem Hamiltonian
      for (int j = 0; j < n_qubits; ++j) {

        x<cudaq::ctrl>(q[j], q[(j + 1) % n_qubits]);
        rz(2.0 * theta[i], q[(j + 1) % n_qubits]);
        x<cudaq::ctrl>(q[j], q[(j + 1) % n_qubits]);
      }

      for (int j = 0; j < n_qubits; ++j) {
        // Mixer Hamiltonian
        rx(2.0 * theta[i + n_layers], q[j]);
      }
    }
  }
};

int main() {

  using namespace cudaq::spin;

  cudaq::set_random_seed(13); // set for repeatability

  // Problem Hamiltonian
  const cudaq::spin_op Hp = 0.5 * z(0) * z(1) + 0.5 * z(1) * z(2) +
                            0.5 * z(0) * z(3) + 0.5 * z(2) * z(3);

  // Problem parameters
  const int n_qubits = 4;
  const int n_layers = 2;
  const int n_params = 2 * n_layers;

  // Instantiate the optimizer
  cudaq::optimizers::cobyla optimizer; // gradient-free COBYLA

  // Set initial values for the optimization parameters
  optimizer.initial_parameters = cudaq::random_vector(
      -M_PI / 8.0, M_PI / 8.0, n_params, std::mt19937::default_seed);

  // Call the optimizer
  auto [opt_val, opt_params] = cudaq::vqe(
      ansatz{}, Hp, optimizer, n_params, [&](std::vector<double> params) {
        return std::make_tuple(params, n_qubits, n_layers);
      });

  // Print the optimized value and the parameters
  printf("Optimal value = %.16lf\n", opt_val);
  printf("Optimal params = (%.16lf, %.16lf, %.16lf, %.16lf) \n", opt_params[0],
         opt_params[1], opt_params[2], opt_params[3]);

  // Sample the circuit using optimized parameters
  auto counts = cudaq::sample(ansatz{}, opt_params, n_qubits, n_layers);

  // Dump the states and the counts
  counts.dump();

  return 0;
}
