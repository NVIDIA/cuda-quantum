// Compile and run with:
// ```
// nvq++ orca_mqpu.cpp --target orca --orca-url \
// "http://localhost:3035,http://localhost:3037" -o out.x && ./out.x
// ```
// See accompanying example `orca.cpp` for detailed explanation.

#include "cudaq.h"
#include "cudaq/orca.h"

#include <fstream>
#include <iostream>

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

  auto &platform = cudaq::get_platform();
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);

  // A time-bin boson sampling experiment
  std::vector<std::size_t> input_state{1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<std::size_t> loop_lengths{1, 1};
  std::size_t sum_loop_lengths{std::accumulate(
      loop_lengths.begin(), loop_lengths.end(), static_cast<std::size_t>(0))};
  const std::size_t n_loops = loop_lengths.size();
  const std::size_t n_modes = input_state.size();
  const std::size_t n_beam_splitters = n_loops * n_modes - sum_loop_lengths;
  std::vector<double> bs_angles(n_beam_splitters);
  linear_spaced_vector(bs_angles, M_PI / 8, M_PI / 3, n_beam_splitters);
  int n_samples{10000};

  std::cout << "Submitting to ORCA Server asynchronously" << std::endl;
  std::vector<cudaq::async_sample_result> countFutures;
  for (std::size_t i = 0; i < num_qpus; i++) {
    countFutures.emplace_back(cudaq::orca::sample_async(
        input_state, loop_lengths, bs_angles, n_samples, i));
  }

  for (auto &counts : countFutures) {
    counts.get().dump();
  }
  return 0;
}
