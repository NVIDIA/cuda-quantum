// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

template <std::size_t N>
struct ghz {
  auto operator()() __qpu__ {

    // Compile-time sized array like std::array
    cudaq::qarray<N> q;
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mx(q);
  }
};

int main() {

  auto kernel = ghz<10>{};
  auto counts = cudaq::sample(kernel);

  if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
    counts.dump();

    // Fine grain access to the bits and counts
    for (auto &[bits, count] : counts) {
      printf("Observed: %s, %lu\n", bits.data(), count);
    }
  }

  return 0;
}


// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ghz{{.*}}{
// CHECK:     %[[MEASOUT:.*]] = quake.mx %[[VAL_0:.*]] : (!quake.veq<10>) -> !cc.stdvec<!quake.measure> 