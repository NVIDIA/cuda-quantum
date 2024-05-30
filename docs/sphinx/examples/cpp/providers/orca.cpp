// Compile and run with:
// ```
// nvq++ --target orca --orca-url http://localhost:8080/sample orca.cpp -o out.x
// && ./out.x
// ```

#include "cudaq/orca.h"
#include "cudaq.h"

int main() {

  std::vector<double> bs_angles{
      M_PI / 3,
      M_PI / 6,
  };
  std::vector<double> ps_angles{
      M_PI / 4,
      M_PI / 5,
  };

  std::vector<std::size_t> input_state{1, 1, 1};
  std::vector<std::size_t> loop_lengths{1};

  int n_samples{10000};

  auto counts = cudaq::orca::sample(bs_angles, ps_angles, input_state,
                                    loop_lengths, n_samples);
  counts.dump();

  return 0;
}