
// RUN: cudaq-quake %s --verify --std=quake1
// RUN: cudaq-quake %s --verify --std=quake2

#include <cudaq.h>

struct InnerKernel {
  void operator()() __qpu__ {
    // empty kernel
  }
};

struct OuterKernel {
  void operator()() __qpu__ {
    InnerKernel inner;
    inner(); // expected-error {{calling a QPU kernel from another QPU kernel is not allowed}}
  }
};

int main() {
  OuterKernel ok;
  ok();
}
