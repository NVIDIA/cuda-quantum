// Compile and run with:
// ```
// nvq++ expectation_values.cpp -o d2.x && ./d2.x
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>

// The example here shows a simple use case for the `cudaq::observe`
// function in computing expected values of provided spin_ops.

struct ansatz {
  auto operator()(double theta) __qpu__ {
    cudaq::qreg q(2);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};

int main() {

  // Build up your spin op algebraically
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  // Observe takes the kernel, the spin_op, and the concrete
  // parameters for the kernel
  double energy = cudaq::observe(ansatz{}, h, .59);
  printf("Energy is %lf\n", energy);
  return 0;
}
