// Compile and run with:
// ```
// nvq++ --target photonics photonics.cpp
// ./a.out
// ```

#include "cudaq/photonics.h"
#include "cudaq.h"

struct photonicsKernel {
  void operator()() __qpu__ {
    cudaq::qvector<3> qutrits(2);
    plus(qutrits[0]);
    plus(qutrits[1]);
    plus(qutrits[1]);
    mz(qutrits);
  }
};

int main() {

  auto counts = cudaq::sample(photonicsKernel{});
  for (auto &[k, v] : counts) {
    printf("Result / Count = %s : %lu\n", k.c_str(), v);
  }
}