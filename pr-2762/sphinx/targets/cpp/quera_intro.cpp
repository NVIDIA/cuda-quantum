// Compile and run with:
// ```
// nvq++ --target quera quera_intro.cpp -o out.x
// ./out.x
// ```
// Assumes a valid set of credentials have been stored.

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <cmath>
#include <map>
#include <vector>

// This example illustrates how to use QuEra's Aquila device on Braket with
// CUDA-Q. It is a CUDA-Q implementation of the getting started materials for
// Braket available here:
// https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/analog_hamiltonian_simulation/01_Introduction_to_Aquila.ipynb

int main() {
  // Topology initialization
  const double separation = 5e-6;
  const double block_separation = 15e-6;
  const int k_max = 5;
  const int m_max = 5;

  std::vector<std::pair<double, double>> register_sites;

  for (int k = 0; k < k_max; ++k) {
    for (int m = 0; m < m_max; ++m) {
      register_sites.push_back(
          std::make_pair(block_separation * m,
                         block_separation * k + separation / std::sqrt(3)));

      register_sites.push_back(std::make_pair(
          block_separation * m - separation / 2,
          block_separation * k - separation / (2 * std::sqrt(3))));

      register_sites.push_back(std::make_pair(
          block_separation * m + separation / 2,
          block_separation * k - separation / (2 * std::sqrt(3))));
    }
  }

  // Simulation Timing
  const double omega_const = 1.5e7;                     // rad/sec
  const double time_ramp = 5e-8;                        // seconds
  const double time_plateau = 7.091995761561453e-08;    // seconds
  const double time_max = 2 * time_ramp + time_plateau; // seconds

  std::vector<std::complex<double>> steps = {
      0.0, time_ramp, time_ramp + time_plateau, time_max};
  cudaq::schedule schedule(steps, {"t"}, {});

  // Rydberg Hamiltonian with trapezoidal omega
  auto omega = cudaq::scalar_operator(
      [time_ramp, time_plateau, time_max,
       omega_const](const std::unordered_map<std::string, std::complex<double>>
                        &parameters) {
        double t = std::real(parameters.at("t"));
        double slope = omega_const / time_ramp;
        double y_intercept = slope * time_max;

        if (t > 0 && t < time_ramp + time_plateau) {
          return std::complex<double>(slope * t, 0.0);
        } else if (t > time_ramp && t < time_max) {
          return std::complex<double>(omega_const, 0.0);
        } else if (t > time_ramp + time_plateau && t < time_max) {
          return std::complex<double>((-slope * t) + y_intercept, 0.0);
        }
        return std::complex<double>(0.0, 0.0);
      });
  auto phi = cudaq::scalar_operator(0.0);
  auto delta = cudaq::scalar_operator(0.0);

  auto hamiltonian =
      cudaq::rydberg_hamiltonian(register_sites, omega, phi, delta);

  // Evolve the system
  auto result = cudaq::evolve(hamiltonian, schedule);
  result.sampling_result->dump();

  return 0;
}
