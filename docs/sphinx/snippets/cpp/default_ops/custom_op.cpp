#include <cudaq.h>

CUDAQ_REGISTER_OPERATION(custom_h, 1, 0,
                         {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2})

CUDAQ_REGISTER_OPERATION(custom_x, 1, 0, {0, 1, 1, 0})

__qpu__ void bell_pair() {
  cudaq::qubit q, r;
  custom_h(q);
  custom_x<cudaq::ctrl>(q, r);
}

int main() {
  auto counts = cudaq::sample(bell_pair);
  for (auto &[bits, count] : counts) {
    printf("%s\n", bits.data());
  }

  return 0;
}
