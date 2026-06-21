#include <cudaq.h>

#include <cstdlib>

struct parametric {
  void operator()() __qpu__ {
    cudaq::qvector q(3);
    ry(3.141592653589793, q[0]);
    rx(3.141592653589793, q[1]);
    rz(1.5707963267948966, q[2]);
    x<cudaq::ctrl>(q[0], q[2]);
    mz(q);
  }
};

int main(int argc, char **argv) {
  const int shots = argc > 1 ? std::atoi(argv[1]) : 100;
  auto counts = cudaq::sample(shots, parametric{});
  counts.dump();
  return 0;
}
