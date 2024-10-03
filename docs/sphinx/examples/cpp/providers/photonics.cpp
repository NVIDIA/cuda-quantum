// Compile and run with:
// ```
// nvq++ --target photonics-cpu photonics.cpp
// ./a.out
// ```

#include "cudaq/photonics.h"
#include "cudaq.h"

struct photonicsKernel {
  void operator()() __qpu__ {
    cudaq::qvector<3> qumodes(2);
    plus(qumodes[0]);
    plus(qumodes[1]);
    plus(qumodes[1]);
    mz(qumodes);
  }
};

int main() {

  auto counts = cudaq::sample(photonicsKernel{});
  for (auto &[k, v] : counts) {
    printf("Result : Count = %s : %lu\n", k.c_str(), v);
  }

  auto state = cudaq::get_state(photonicsKernel{});
  state.dump();
}