// Compile and run with:
// ```
// nvq++ noise_amplitude_damping.cpp --target density-matrix-cpu -o dyn.x
// --target nvidia && ./dyn.x
// ```
//
// Note: You must set the target to a density matrix backend for the noise
// to successfully impact the system.

#include <cudaq.h>

// CUDA Quantum supports several different models of noise. In this case,
// we will examine the modeling of energy dissipation within our system
// via environmental interactions. The result of this "amplitude damping"
// is to return the qubit to the |0> state with a user-specified probability.

int main() {

  // We will begin by defining an empty noise model that we will add
  // our damping channel to.
  cudaq::noise_model noise;

  // Amplitude damping channel with `1.0` probability of the qubit
  // decaying to the ground state.
  cudaq::amplitude_damping_channel ad(1.);

  // We will apply this channel to any Hadamard gate on the qubit.
  // Meaning, after each Hadamard on the qubit, there will be a
  // probability of `1.0` that the qubit decays back to ground.
  noise.add_channel<cudaq::types::h>({0}, ad);

  // The Hadamard gate here will bring the qubit to `1/sqrt(2) (|0> + |1>)`,
  // where it will remain with a probability of `1 - p = 0.0`.
  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  };

  // Now let's set the noise and we're ready to run the simulation!
  cudaq::set_noise(noise);

  // Our results should show all measurements in the |0> state, indicating
  // that the noise has successfully impacted the system.
  auto noisy_counts = cudaq::sample(kernel);
  noisy_counts.dump();

  // To confirm this, we can run the simulation again without noise.
  // The qubit will now have a 50/50 mix of measurements between
  // |0> and |1>.
  cudaq::unset_noise();
  auto noiseless_counts = cudaq::sample(kernel);
  noiseless_counts.dump();
}