#include <cudaq.h>

#include <cstdlib>

struct phase_kickback {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    x(q[1]);
    h(q[0]);
    r1<cudaq::ctrl>(3.141592653589793, q[0], q[1]);
    h(q[0]);
    mz(q);
  }
};

int main(int argc, char **argv) {
  const int shots = argc > 1 ? std::atoi(argv[1]) : 100;
  auto counts = cudaq::sample(shots, phase_kickback{});
  counts.dump();
  return 0;
}
