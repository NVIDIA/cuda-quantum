#include <cudaq.h>

#include <cstdlib>

struct bell {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    mz(q);
  }
};

int main(int argc, char **argv) {
  const int shots = argc > 1 ? std::atoi(argv[1]) : 100;
  auto counts = cudaq::sample(shots, bell{});
  counts.dump();
  return 0;
}
