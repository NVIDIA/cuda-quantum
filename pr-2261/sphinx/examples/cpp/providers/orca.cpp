// Compile and run with:
// ```
// nvq++ --target orca --orca-url $ORCA_ACCESS_URL orca.cpp -o out.x && ./out.x
// ```
// To use the ORCA Computing target you will need to set the ORCA_ACCESS_URL
// environment variable or pass the URL to the `--orca-url` flag.

#include "cudaq/orca.h"
#include "cudaq.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

// define helper function to generate linear spaced vectors
template <typename T>
void linear_spaced_vector(std::vector<T> &xs, T min, T max, std::size_t N) {
  T h = (max - min) / static_cast<T>(N - 1);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = min; x != xs.end(); ++x, val += h) {
    *x = val;
  }
}

int main() {
  using namespace std::this_thread;     // sleep_for, sleep_until
  using namespace std::chrono_literals; // `ns`, `us`, `ms`, `s`, `h`, etc.

  // A time-bin boson sampling experiment: An input state of 4 indistinguishable
  // photons mixed with 4 vacuum states across 8 time bins (modes) enter the
  // time bin interferometer (TBI). The interferometer is composed of two loops
  // each with a beam splitter (and optionally with a corresponding phase
  // shifter). Each photon can either be stored in a loop to interfere with the
  // next photon or exit the loop to be measured. Since there are 8 time bins
  // and 2 loops, there is a total of 14 beam splitters (and optionally 14 phase
  // shifters) in the interferometer, which is the number of controllable
  // parameters.

  // half of 8 time bins is filled with a single photon and the other half is
  // filled with the vacuum state (empty)
  std::vector<std::size_t> input_state{1, 0, 1, 0, 1, 0, 1, 0};

  // The time bin interferometer in this example has two loops, each of length 1
  std::vector<std::size_t> loop_lengths{1, 1};

  // helper variables to calculate the number of beam splitters and phase
  // shifters needed in the TBI
  std::size_t sum_loop_lengths{std::accumulate(
      loop_lengths.begin(), loop_lengths.end(), static_cast<std::size_t>(0))};
  const std::size_t n_loops = loop_lengths.size();
  const std::size_t n_modes = input_state.size();
  const std::size_t n_beam_splitters = n_loops * n_modes - sum_loop_lengths;

  // beam splitter angles (created as a linear spaced vector of angles)
  std::vector<double> bs_angles(n_beam_splitters);
  linear_spaced_vector(bs_angles, M_PI / 8, M_PI / 3, n_beam_splitters);

  // Optionally, we can also specify the phase shifter angles (created as a
  // linear spaced vector of angles), if the system includes phase shifters
  // ```
  // std::vector<double> ps_angles(n_beam_splitters);
  // linear_spaced_vector(ps_angles, M_PI / 6, M_PI / 3, n_beam_splitters);
  // ```

  // we can also set number of requested samples
  int n_samples{10000};

  // Submit to ORCA synchronously (e.g., wait for the job result to be
  // returned before proceeding with the rest of the execution).
  std::cout << "Submitting to ORCA Server synchronously" << std::endl;
  auto counts =
      cudaq::orca::sample(input_state, loop_lengths, bs_angles, n_samples);

  // Print the results
  counts.dump();

  // If the system includes phase shifters, the phase shifter angles can be
  // included in the call

  // ```
  // auto counts = cudaq::orca::sample(input_state, loop_lengths, bs_angles,
  //                                   ps_angles, n_samples);
  // ```

  // Alternatively we can submit to ORCA asynchronously (e.g., continue
  // executing code in the file until the job has been returned).
  std::cout << "Submitting to ORCA Server asynchronously" << std::endl;
  auto async_results = cudaq::orca::sample_async(input_state, loop_lengths,
                                                 bs_angles, n_samples);

  // Can write the future to file:
  {
    std::ofstream out("saveMe.json");
    out << async_results;
  }

  // Then come back and read it in later.
  cudaq::async_result<cudaq::sample_result> readIn;
  std::ifstream in("saveMe.json");
  in >> readIn;

  sleep_for(200ms); // wait for the job to be processed
  // Get the results of the read in future.
  auto async_counts = readIn.get();
  async_counts.dump();

  return 0;
}