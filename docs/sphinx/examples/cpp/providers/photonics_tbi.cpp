// Compile and run with:
// ```
// nvq++ --target photonics photonics_tbi.cpp -o tbi.x
// CUDAQ_LOG_LEVEL=info ./tbi.x
// ```

#include "cudaq.h"
#include "cudaq/photonics.h"

#include <iostream>

// Global variables

static const std::size_t zero{0};
static const std::size_t one{1};

static constexpr std::size_t n_modes{4};
static constexpr std::array<std::size_t, n_modes> input_state{2, 1, 3, 1};

static constexpr std::size_t d{
    std::accumulate(input_state.begin(), input_state.end(), one)};

static constexpr std::size_t n_loops{2};
static constexpr std::array<std::size_t, n_loops> loop_lengths{1, 2};

const std::size_t sum_loop_lenghts{
    std::accumulate(loop_lengths.begin(), loop_lengths.end(), zero)};

static constexpr std::size_t n_beamsplitters =
    n_loops * n_modes - sum_loop_lenghts;

struct TBIParameters {
  std::vector<double> bs_angles;
  std::vector<double> ps_angles;

  std::array<std::size_t, ::n_modes> input_state{::input_state};
  std::array<std::size_t, ::n_loops> loop_lengths{::loop_lengths};

  int n_samples = 1000000;
};

struct TBI {
  auto operator()(TBIParameters const parameters) __qpu__ {
    auto n_modes = ::n_modes;
    auto input_state = ::input_state;
    const auto d = ::d;
    auto loop_lengths = ::loop_lengths;

    auto bs_angles = parameters.bs_angles;
    auto ps_angles = parameters.ps_angles;

    cudaq::qvector<d> quds(n_modes); // |00...00> d-dimensions
    for (std::size_t i = 0; i < n_modes; i++) {
      for (std::size_t j = 0; j < input_state[i]; j++) {
        plus(quds[i]); // setting to  |input_state>
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
  std::vector<double> bs_angles(n_beamsplitters);
  std::vector<double> ps_angles(n_beamsplitters);
  LinearSpacedArray(bs_angles, M_PI / 3, M_PI / 6, n_beamsplitters);
  LinearSpacedArray(ps_angles, M_PI / 3, M_PI / 5, n_beamsplitters);

  const TBIParameters parameters{
      bs_angles,
      ps_angles,
  };

  auto counts = cudaq::sample(1000000, TBI{}, parameters);

  std::map<std::string, std::size_t> ordered_counts(counts.begin(),
                                                    counts.end());
  for (auto &[k, v] : ordered_counts) {
    std::cout << "Sample " << k << " : " << v << std::endl;
  }

  return 0;
}