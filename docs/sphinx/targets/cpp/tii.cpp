// 
// Available projects and devices are show in the dashboard:
// https://q-cloud.tii.ae/projects/
//
// The authentication key must be stored (or exported) as enivronment variable:
// ```
// export TII_API_TOKEN="you-tii-token"
// ```
//
// Compile and run with:
// ```
// nvq++ --target tii --tii-device tii-sim \
// --tii-project <tii-project-name> tii.cpp
// ./a.out
// ```

#include <cudaq.h>
#include <fstream>

struct bell_state {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    mz(q);
  }
};

int main() {
  // Submit to `tii` synchronously.
  auto result_counts = cudaq::sample(bell_state{});
  result_counts.dump();
}
