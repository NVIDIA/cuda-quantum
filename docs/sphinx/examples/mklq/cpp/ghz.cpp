#include <cudaq.h>

#include <cstdlib>

struct ghz {
  void operator()() __qpu__ {
    cudaq::qvector q(3);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[1], q[2]);
    mz(q);
  }
};

int main(int argc, char **argv) {
  const int shots = argc > 1 ? std::atoi(argv[1]) : 100;
  auto counts = cudaq::sample(shots, ghz{});
  counts.dump();
  return 0;
}
