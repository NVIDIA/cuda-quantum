// Compile and run with:
// ```
// nvq++ noise_callback.cpp --target density-matrix-cpu -o dyn.x
// && ./dyn.x
// ```
//
// Note: You must set the target to a density matrix backend for the noise
// to successfully impact the system.

#include <cudaq.h>
#include <iostream>

// CUDA-Q supports several different models of noise. In this
// case, we will examine the dynamic noise channel specified as a callback
// function.

int main() {

  // We will begin by defining an empty noise model that we will add
  // our channel to.
  cudaq::noise_model noise;
  //  Noise model callback function
  const auto rx_noise = [](const auto &qubits,
                           const auto &params) -> cudaq::kraus_channel {
    // Model a pulse-length based rotation gate:
    // the bigger the angle, the longer the pulse, i.e., more amplitude damping.
    auto angle = params[0];
    // Normalize the angle into the [0, 2*pi] range
    while (angle > 2. * M_PI)
      angle -= 2. * M_PI;

    while (angle < 0)
      angle += 2. * M_PI;
    // Damping rate is linearly proportional to the angle
    const auto damping_rate = angle / (2. * M_PI);
    std::cout << "Angle = " << params[0]
              << ", amplitude damping rate = " << damping_rate << "\n";
    return cudaq::amplitude_damping_channel(damping_rate);
  };

  // Bind the noise model callback function to the `rx` gate
  noise.add_channel<cudaq::types::rx>(rx_noise);

  auto kernel = [](double angle) __qpu__ {
    cudaq::qubit q;
    rx(angle, q);
    mz(q);
  };

  // Now let's set the noise and we're ready to run the simulation!
  cudaq::set_noise(noise);

  // Our results should show measurements in both the |0> and |1> states,
  // indicating that the noise has successfully impacted the system. Note: a
  // `rx(pi)` is equivalent to a Pauli X gate, and thus, it should be in the |1>
  // state if no noise is present.
  auto noisy_counts = cudaq::sample(kernel, M_PI);
  std::cout << "Noisy result:\n";
  noisy_counts.dump();

  // To confirm this, we can run the simulation again without noise.
  cudaq::unset_noise();
  auto noiseless_counts = cudaq::sample(kernel, M_PI);
  std::cout << "Noiseless result:\n";
  noiseless_counts.dump();
  return 0;
}
