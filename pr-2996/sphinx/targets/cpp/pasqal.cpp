// Compile and run with:
// ```
// nvq++ --target pasqal pasqal.cpp -o out.x
// ./out.x
// ```
// Assumes a valid set of credentials (`PASQAL_AUTH_TOKEN`, `PASQAL_PROJECT_ID`)
// have been set.

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <cmath>
#include <map>
#include <vector>

// This example illustrates how to use `Pasqal's` EMU_MPS emulator over
// `Pasqal's` cloud via CUDA-Q. Contact Pasqal at help@pasqal.com or through
// https://community.pasqal.com for assistance.

int main() {
  // Topology initialization
  const double a = 5e-6;
  std::vector<std::pair<double, double>> register_sites;
  register_sites.push_back(std::make_pair(a, 0.0));
  register_sites.push_back(std::make_pair(2 * a, 0.0));
  register_sites.push_back(std::make_pair(3 * a, 0.0));

  // Simulation Timing
  const double time_ramp = 0.000001; // seconds
  const double time_max = 0.000003;  // seconds
  const double omega_max = 1000000;  // rad/sec
  const double delta_end = 1000000;
  const double delta_start = 0.0;

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
  auto result = cudaq::evolve(hamiltonian, schedule, 100);
  result.sampling_result->dump();

  return 0;
}
