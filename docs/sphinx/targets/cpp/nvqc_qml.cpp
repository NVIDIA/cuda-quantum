// Compile and run with:
// ```
// nvq++ --target nvqc nvqc_qml.cpp -o out.x
// ./out.x
// ```
// Assumes a valid NVQC API key has been set in the `NVQC_API_KEY` environment
// variable. Please refer to the documentations for information about how to
// attain NVQC API key.

// This example demonstrates the simulation of large parameterized quantum
// circuits with NVQC. These parameterized circuits can be applied to quantum
// machine learning to classify data points, e.g. see
// https://arxiv.org/pdf/1906.07682.pdf.

#include <ctime>
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/optimizers.h>
#include <random>

struct kernel {

  auto operator()(const int n_qubits,
                  const std::vector<double> parameters) __qpu__ {

    cudaq::qvector qubits(n_qubits);
    h(qubits);

    for (size_t i = 0; i < n_qubits; i++) {
      rx(parameters[i], qubits[i]);
    }

    for (size_t i = 0; i < n_qubits; i++) {
      ry(parameters[i + n_qubits], qubits[i]);
    }

    h(qubits);

    for (size_t i = 0; i < n_qubits; i++) {
      rz(parameters[i + n_qubits * 2], qubits[i]);
    }

    for (size_t i = 0; i < n_qubits; i += 2) {
      cx(qubits[i], qubits[i + 1]);
    }

    for (size_t i = 0; i < n_qubits; i++) {
      rz(parameters[i + n_qubits * 2], qubits[i]);
    }

    for (size_t i = 0; i < n_qubits; i += 2) {
      cy(qubits[i], qubits[i + 1]);
    }

    for (size_t i = 0; i < n_qubits; i++) {
      ry(parameters[i + n_qubits], qubits[i]);
    }

    for (size_t i = 0; i < n_qubits; i += 2) {
      cz(qubits[i], qubits[i + 1]);
    }

    x(qubits);
    y(qubits);
    h(qubits);
  }
};

std::vector<double> initial_parameters(int n_parameters, int seed) {

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  std::vector<double> parameters(n_parameters);

  for (size_t i = 0; i < n_parameters; i++) {
    parameters[i] = distribution(generator);
  }
  return parameters;
}

int main() {

  const int n_qubits = 26;
  const int n_parameters = 3 * n_qubits;
  std::vector<double> parameters = initial_parameters(n_parameters, 13);
  auto h = cudaq::spin::z(0);

  auto start = clock();
  auto exp_val = cudaq::observe(kernel{}, h, n_qubits, parameters);
  auto end = clock();

  printf("Expectation Value: %f \n", exp_val.expectation());
  printf("Runtime: %fs \n", float(end - start) / float(CLOCKS_PER_SEC));

  return 0;
}
