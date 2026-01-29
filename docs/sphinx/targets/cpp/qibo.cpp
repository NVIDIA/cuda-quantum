// Compile and run with:
// ```
// nvq++ --library-mode --target qibo qibo.cpp
// ./a.out
// ```

#include <cudaq.h>
#include <fstream>

struct bell_state {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    auto result = mz(q);
  }
};

int main() {
  // OR: Submit to Qibo synchronously.
  auto result_counts = cudaq::sample(bell_state{});
  result_counts.dump();
}
