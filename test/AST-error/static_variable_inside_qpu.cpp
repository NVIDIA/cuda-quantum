// RUN: cudaq-quake %s --verify --std=quake1
// RUN: cudaq-quake %s --verify --std=quake2

#include <cudaq.h>

struct StaticInKernel {
  void operator()() __qpu__ {
    static int counter = 0; // expected-error {{static variables are not allowed inside QPU kernels}}
    counter++;
  }
};

int main() {
  StaticInKernel k;
  k();
}