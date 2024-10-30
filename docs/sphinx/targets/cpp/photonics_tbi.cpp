// Compile and run with:
// ```
// nvq++ --target orca-photonics photonics_tbi.cpp -o tbi.x
// ./tbi.x
// ```

#include "cudaq.h"
#include "cudaq/photonics.h"

#include <iostream>

// Global variables
static const std::size_t one{1};

static constexpr std::size_t n_modes{4};
static constexpr std::array<std::size_t, n_modes> input_state{2, 1, 3, 1};

static constexpr std::size_t d{
    std::accumulate(input_state.begin(), input_state.end(), one)};

struct TBI {
  auto operator()(std::vector<double> const &bs_angles,
                  std::vector<double> const &ps_angles,
                  std::vector<std::size_t> const &input_state,
                  std::vector<std::size_t> const &loop_lengths) __qpu__ {
    auto n_modes = ::n_modes;
    const auto d = ::d;

    cudaq::qvector<d> quds(n_modes); // |00...00> d-dimensions
    for (std::size_t i = 0; i < n_modes; i++) {
      for (std::size_t j = 0; j < input_state[i]; j++) {
        plus(quds[i]); // setting to |input_state>
      }
    }

    std::size_t c = 0;
    for (std::size_t ll : loop_lengths) {
      for (std::size_t i = 0; i < (n_modes - ll); i++) {
        beam_splitter(quds[i], quds[i + ll], bs_angles[c]);
        phase_shift(quds[i], ps_angles[c]);
        c++;
      }
    }
    mz(quds);
  }
};

template <typename T>
void LinearSpacedArray(std::vector<T> &xs, T min, T max, std::size_t N) {
  T h = (max - min) / static_cast<T>(N - 1);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = min; x != xs.end(); ++x, val += h) {
    *x = val;
  }
}

int main() {
  std::size_t n_loops{2};
  std::vector<std::size_t> loop_lengths{1, 2};
  std::vector<std::size_t> input_state(std::begin(::input_state),
                                       std::end(::input_state));

  const std::size_t zero{0};
  std::size_t sum_loop_lenghts{
      std::accumulate(loop_lengths.begin(), loop_lengths.end(), zero)};

  std::size_t n_beam_splitters = n_loops * ::n_modes - sum_loop_lenghts;

  std::vector<double> bs_angles(n_beam_splitters);
  std::vector<double> ps_angles(n_beam_splitters);

  LinearSpacedArray(bs_angles, M_PI / 3, M_PI / 6, n_beam_splitters);
  LinearSpacedArray(ps_angles, M_PI / 3, M_PI / 5, n_beam_splitters);

  auto counts = cudaq::sample(1000000, TBI{}, bs_angles, ps_angles, input_state,
                              loop_lengths);

  for (auto &[k, v] : counts) {
    std::cout << k << ":" << v << " ";
  }
  std::cout << std::endl;
  return 0;
}