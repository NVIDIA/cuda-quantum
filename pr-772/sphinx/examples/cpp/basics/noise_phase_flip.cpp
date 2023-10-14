// Compile and run with:
// ```
// nvq++ noise_phase_flip.cpp --target density-matrix-cpu -o dyn.x --target
// nvidia && ./dyn.x
// ```
//
// Note: You must set the target to a density matrix backend for the noise
// to successfully impact the system.

#include <cudaq.h>

// CUDA Quantum supports several different models of noise. In this
// case, we will examine the modeling of decoherence of the qubit phase.
// This will occur from "phase flip" errors, wherein the qubit has a
// user-specified probability of undergoing a Z-180 rotation.

int main() {

  // We will begin by defining an empty noise model that we will add
  // our phase flip channel to.
  cudaq::noise_model noise;

  // Phase flip channel with `1.0` probability of the qubit
  // undergoing a phase rotation of 180 degrees (π).
  cudaq::phase_flip_channel pf(1.);
  // We will apply this channel to any Z gate on the qubit.
  // Meaning, after each Z gate on qubit 0, there will be a
  // probability of `1.0` that the qubit undergoes an extra
  // Z rotation.
  noise.add_channel<cudaq::types::z>({0}, pf);

  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    // Place qubit in superposition state.
    h(q);
    // Rotate on Z by 180 degrees.
    z(q);
    // Apply another Hadamard.
    h(q);
    mz(q);
  };

  // Now let's set the noise and we're ready to run the simulation!
  cudaq::set_noise(noise);

  // With noise, our Z-gate will effectively cancel out due
  // to the presence of a phase flip error on the gate with a
  // probability of `1.0`. This will put us back in the |0> state.
  auto noisy_counts = cudaq::sample(kernel);
  noisy_counts.dump();

  // To confirm this, we can run the simulation again without noise.
  // Without noise, we'd expect the qubit to end in the |1> state due
  // to the phase rotation between the two Hadamard gates.
  cudaq::unset_noise();
  auto noiseless_counts = cudaq::sample(kernel);
  noiseless_counts.dump();
}
