// Compile and run with:
// ```
// nvq++ --target quera quera_basic.cpp -o out.x
// ./out.x
// ```
// Assumes a valid set of credentials have been stored.

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/schedule.h"
#include <cmath>
#include <map>
#include <vector>

// NOTE: QuEra Aquila system is available via Amazon Braket.
// Credentials must be set before running this program.
// Amazon Braket costs apply.

// This example illustrates how to use QuEra's Aquila device on Braket with
// CUDA-Q. It is a CUDA-Q implementation of the getting started materials for
// Braket available here:
// https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html

int main() {
  // Topology initialization
  const double a = 5.7e-6;
  std::vector<std::pair<double, double>> register_sites;

  auto make_coord = [a](double x, double y) {
    return std::make_pair(x * a, y * a);
  };

  register_sites.push_back(make_coord(0.5, 0.5 + 1.0 / std::sqrt(2)));
  register_sites.push_back(make_coord(0.5 + 1.0 / std::sqrt(2), 0.5));
  register_sites.push_back(make_coord(0.5 + 1.0 / std::sqrt(2), -0.5));
  register_sites.push_back(make_coord(0.5, -0.5 - 1.0 / std::sqrt(2)));
  register_sites.push_back(make_coord(-0.5, -0.5 - 1.0 / std::sqrt(2)));
  register_sites.push_back(make_coord(-0.5 - 1.0 / std::sqrt(2), -0.5));
  register_sites.push_back(make_coord(-0.5 - 1.0 / std::sqrt(2), 0.5));
  register_sites.push_back(make_coord(-0.5, 0.5 + 1.0 / std::sqrt(2)));

  // Simulation Timing
  const double time_max = 4e-6;   // seconds
  const double time_ramp = 1e-7;  // seconds
  const double omega_max = 6.3e6; // rad/sec
  const double delta_start = -5 * omega_max;
  const double delta_end = 5 * omega_max;

  std::vector<std::complex<double>> steps = {0.0, time_ramp,
                                             time_max - time_ramp, time_max};
  cudaq::schedule schedule(steps, {"t"}, {});

  // Basic Rydberg Hamiltonian
  auto omega = cudaq::scalar_operator(
      [time_ramp, time_max,
       omega_max](const std::unordered_map<std::string, std::complex<double>>
                      &parameters) {
        double t = std::real(parameters.at("t"));
        return std::complex<double>(
            (t > time_ramp && t < time_max) ? omega_max : 0.0, 0.0);
      });

  auto phi = cudaq::scalar_operator(0.0);

  auto delta = cudaq::scalar_operator(
      [time_ramp, time_max, delta_start,
       delta_end](const std::unordered_map<std::string, std::complex<double>>
                      &parameters) {
        double t = std::real(parameters.at("t"));
        return std::complex<double>(
            (t > time_ramp && t < time_max) ? delta_end : delta_start, 0.0);
      });

  auto hamiltonian =
      cudaq::rydberg_hamiltonian(register_sites, omega, phi, delta);

  // Evolve the system
  auto result = cudaq::evolve_async(hamiltonian, schedule, 10).get();
  result.sampling_result->dump();

  return 0;
}
