// Compile and run with:
// ```
// nvq++ test_photonics.cpp --target photonics
// ./a.out
// ```

#include "cudaq.h"
#include "cudaq/photonics.h"

struct photonicsKernel {
  void operator()() __qpu__ {
    cudaq::qreg<cudaq::dyn, 3> qutrits(2);
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