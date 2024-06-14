// Compile and run with:
// ```
// nvq++ --target orca --orca-url $ORCA_ACCESS_URL orca.cpp -o out.x
// && ./out.x
// ```
// To use the ORCA target you will need to set the ORCA_ACCESS_URL environment
// variable or pass the url to the --orca-url flag.

// ORCA's PT-Series implement the boson sampling model of quantum computation,
// in which multiple photons are interfered with each other within a network of
// beam splitters, and photon detectors measure where the photons leave this
// network.
// The parameters needed to define the time bin interferometer are the
// beam splitter angles, the phase shifter angles, the input state, the loop
// lengths and optionally the number of samples.
// The input state is the initial state of the photons in the time bin
// interferometer, the left-most entry corresponds to the first mode entering
// the loop.
// The loop lengths are the the lengths of the different loops in the time bin
// interferometer.

#include "cudaq/orca.h"
#include "cudaq.h"

int main() {

  // A time-bin boson sampling experiment: An input state of 3 indistinguishable
  // photons across 3 time bins (modes) entering the time bin interferometer.
  // The interferometer is composed of one loop and beam splitter with its
  // correspondent phase shifter. Each photon can either be stored in the loop
  // to interfere with the next photon or exit the loop to be measured.
  // Since there are 3 time time bins, there are 2 beam splitters and 2 phase
  // shifters in the interferometer.

  // beam splitter angles
  std::vector<double> bs_angles{
      M_PI / 3,
      M_PI / 6,
  };

  // phase shifter angles
  std::vector<double> ps_angles{
      M_PI / 4,
      M_PI / 5,
  };

  // all time bins are filled witha a single photon
  std::vector<std::size_t> input_state{1, 1, 1};

  // The time bin interferometer in this example has only one loop of length 1
  std::vector<std::size_t> loop_lengths{1};

  int n_samples{10000};

  // Submit to ORCA synchronously (e.g., wait for the job result to be returned
  // before proceeding).
  auto counts = cudaq::orca::sample(bs_angles, ps_angles, input_state,
                                    loop_lengths, n_samples);
  counts.dump();

  return 0;
}