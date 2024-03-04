#include <cudaq.h>
#include <cudaq/optimizers.h>
#include <cudaq/algorithm.h>
#include <cudaq/algorithms/gradients/central_difference.h>


using namespace cudaq::spin;


// We define the variational quantum kernel that we'd like
// to use as an ansatz.
// Creating a kernel that takes a float as a function argument.
void __qpu__ kernel(double theta) {
  // Allocate 2 qubits.
  cudaq::qvector qvector(2);
  x(qvector[0]);
  // Apply an `ry` gate that is parameterized by our `theta`.
  ry(theta, qvector[1]);
  x<cudaq::ctrl>(qvector[1], qvector[0]);
  // Note: the kernel must not contain measurement instructions.
}

int main() {
  // We begin by defining the spin Hamiltonian for the system that we are working
  // with. This is achieved through the use of `cudaq::spin_op`'s, which allow
  // for the convenient creation of complex Hamiltonians out of Pauli spin operators.
  cudaq::spin_op hamiltonian = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                      .21829 * z(0) - 6.125 * z(1);

  // // Next, we define the variational quantum kernel that we'd like
  // // to use as an ansatz.
  // // Create a kernel that takes a float as a function argument.
  // auto [kernel, theta] = cudaq::make_kernel<double>();
  // // Allocate 2 qubits.
  // auto qvector = kernel.qalloc(2);
  // kernel.x(qvector[0]);
  // // Apply an `ry` gate that is parameterized by our `theta`.
  // kernel.ry(theta, qvector[1]);
  // kernel.x<cudaq::ctrl>(qvector[1], qvector[0]);
  // // Note: the kernel must not contain measurement instructions.

  // Need an argument mapper to indicate to the VQE module which
  // parameters belong to which gates.
  // auto argument_mapper = [](double value) { return value; };

  // The last thing we need is to pick an optimizer from the suite of `cudaq.optimizers`.
  // We can optionally tune this optimizer through its initial parameters, iterations,
  // optimization bounds, etc. before passing it to `cudaq.vqe`.

  cudaq::optimizers::cobyla optimizer;

  auto [energy, angle] = cudaq::vqe(kernel, hamiltonian, optimizer, 1);

  // auto [energy, angle] = optimizer.optimize(1, [&](double theta){
  //   return cudaq::observe(kernel, hamiltonian, theta);
  // });
  printf("<H3> = %lf\n", energy);
}