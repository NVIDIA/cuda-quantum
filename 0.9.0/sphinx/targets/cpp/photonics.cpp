// Compile and run with:
// ```
// nvq++ --target orca-photonics photonics.cpp
// ./a.out
// ```

#include "cudaq/photonics.h"
#include "cudaq.h"

struct photonicsKernel {
  void operator()() __qpu__ {
    cudaq::qvector<3> qumodes(2);
    create(qumodes[0]);
    create(qumodes[1]);
    create(qumodes[1]);
    mz(qumodes);
  }
};

int main() {

  auto counts = cudaq::sample(photonicsKernel{});
  counts.dump();

  auto state = cudaq::get_state(photonicsKernel{});
  state.dump();
}