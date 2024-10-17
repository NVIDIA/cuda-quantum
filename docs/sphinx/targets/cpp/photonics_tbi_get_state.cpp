// Compile and run with:
// ```
// nvq++ --target photonics photonics_tbi_get_state.cpp && ./a.out
// ```

#include <cudaq.h>
#include <cudaq/photonics.h>
#include <iostream>

// Global variables
static constexpr std::size_t one = 1;

static constexpr std::size_t n_modes = 8;
static constexpr std::array<std::size_t, n_modes> input_state = {1, 0, 1, 0,
                                                                 1, 0, 1, 0};

static constexpr std::size_t d =
    std::accumulate(input_state.begin(), input_state.end(), one);

struct TBI {
  auto operator()(std::vector<double> const &bs_angles,
                  std::vector<double> const &ps_angles,
                  std::vector<std::size_t> const &input_state,
                  std::vector<std::size_t> const &loop_lengths) __qpu__ {
    auto n_modes = ::n_modes;
    const auto d = ::d;

    cudaq::qvector<d> qumodes(n_modes); // |00...00> d-dimensions
    for (std::size_t i = 0; i < n_modes; i++) {
      for (std::size_t j = 0; j < input_state[i]; j++) {
        create(qumodes[i]); // setting to |input_state>
      }
    }

    std::size_t c = 0;
    for (std::size_t ll : loop_lengths) {
      for (std::size_t i = 0; i < (n_modes - ll); i++) {
        beam_splitter(qumodes[i], qumodes[i + ll], bs_angles[c]);
        phase_shift(qumodes[i], ps_angles[c]);
        c++;
      }
    }
  }
};

int main() {
  std::size_t n_loops = 2;
  std::vector<std::size_t> loop_lengths = {1, 1};
  std::vector<std::size_t> input_state(std::begin(::input_state),
                                       std::end(::input_state));

  const std::size_t zero = 0;
  std::size_t sum_loop_lenghts{
      std::accumulate(loop_lengths.begin(), loop_lengths.end(), zero)};

  std::size_t n_beam_splitters = n_loops * ::n_modes - sum_loop_lenghts;

  std::vector<double> bs_angles =
      cudaq::linspace(M_PI / 3, M_PI / 6, n_beam_splitters);
  std::vector<double> ps_angles =
      cudaq::linspace(M_PI / 3, M_PI / 5, n_beam_splitters);

  auto state =
      cudaq::get_state(TBI{}, bs_angles, ps_angles, input_state, loop_lengths);

  state.dump();

  return 0;
}