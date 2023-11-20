// Compile and run with:
// ```
// nvq++ noise_readout_error.cpp --target density-matrix-cpu -o readout.x
// && ./readout.x
// ```
//
// Note: You must set the target to a density matrix backend for the noise
// to successfully impact the system.

#include <cudaq.h>

// CUDA Quantum supports several different models of noise. In this
// case, we will examine the modeling of readout error.

int main() {

  // We will begin by defining an empty noise model that we will add
  // our depolarization channel to.
  cudaq::noise_model noise;


  // Readout error model using the simplest constructor. This noise model is that
  // each qubit is treated independently, and with the same probabilities. Here, the
  // two arguments are
  // p0 = p(1|0) = the probability that |0> is measured as |1> and
  // p1 = p(0|1) = the probability that |1> is measure as |0>.
  cudaq::readout_error_model error_model(0.1, 0.2);
  // Two other constructors worth including could be:
  // 1) Independent (but different) qubit readout errors. So a p0,p1 pair for each qubit. and
  // 2) Loading in the full 2^N by 2^N confusion matrix.

  // We will apply the readout error to the noise model
  noise.add_readout_error(error_model);

  // Our kernel will initialize a single qubit in the |0> state,
  // and a second qubit in the |1> state.
  auto kernel = []() __qpu__ {
    cudaq::qubit q(2);
    x(q[1]);
    mz(q);
  };

  // Now let's set the noise and we're ready to run the simulation!
  cudaq::set_noise(noise);

  // With noise, we should find qubit 0 as been measured in the |1> state about 10% of the time,
  // and qubit 1 in the |0> state about 80% of the time.
  auto noisy_counts = cudaq::sample(kernel);
  noisy_counts.dump();

  // To confirm this, we can run the simulation again without noise.
  // Without noise, the qubits should still be in the |01> state.
  cudaq::unset_noise();
  auto noiseless_counts = cudaq::sample(kernel);
  noiseless_counts.dump();
}
