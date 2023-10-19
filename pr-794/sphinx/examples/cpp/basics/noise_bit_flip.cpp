// Compile and run with:
// ```
// nvq++ noise_bit_flip.cpp --target density-matrix-cpu -o dyn.x --target nvidia
// && ./dyn.x
// ```
//
// Note: You must set the target to a density matrix backend for the noise
// to successfully impact the system.

#include <cudaq.h>

// CUDA Quantum supports several different models of noise. In this case,
// we will examine the modeling of decoherence of the qubit state. This
// will occur from "bit flip" errors, wherein the qubit has a user-specified
// probability of undergoing an X-180 rotation.

int main() {

  // We will begin by defining an empty noise model that we will add
  // these decoherence channels to.
  cudaq::noise_model noise;

  // Bit flip channel with `1.0` probability of the qubit flipping 180 degrees.
  cudaq::bit_flip_channel bf(1.);
  // We will apply this channel to any X gate on the qubit, giving each X-gate
  // a probability of `1.0` of undergoing an extra X-gate.
  noise.add_channel<cudaq::types::x>({0}, bf);

  // After the X-gate, the qubit will remain in the |1> state with a probability
  // of `1 - p = 0.0`.
  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    x(q);
    mz(q);
  };

  // Now let's set the noise and we're ready to run the simulation!
  cudaq::set_noise(noise);

  // Our results should show all measurements in the |0> state, indicating
  // that the noise has successfully impacted the system.
  auto noisy_counts = cudaq::sample(kernel);
  noisy_counts.dump();

  // To confirm this, we can run the simulation again without noise.
  // We should now see the qubit in the |1> state.
  cudaq::unset_noise();
  auto noiseless_counts = cudaq::sample(kernel);
  noiseless_counts.dump();
}
