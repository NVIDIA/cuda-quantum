#include <cudaq.h>

#include <cstdlib>

struct clifford_chain {
  void operator()() __qpu__ {
    cudaq::qvector q(4);
    x(q[0]);
    x(q[3]);
    swap(q[0], q[2]);
    h(q[1]);
    z(q[1]);
    h(q[1]);
    x<cudaq::ctrl>(q[2], q[0]);
    z<cudaq::ctrl>(q[0], q[3]);
    s(q[2]);
    sdg(q[2]);
    mz(q);
  }
};

int main(int argc, char **argv) {
  const int shots = argc > 1 ? std::atoi(argv[1]) : 100;
  auto counts = cudaq::sample(shots, clifford_chain{});
  counts.dump();
  return 0;
}
