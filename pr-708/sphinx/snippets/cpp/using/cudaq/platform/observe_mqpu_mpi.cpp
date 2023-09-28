// Compile and run with:
// ```
// nvq++ observe_mqpu_mpi.cpp -o observe_mqpu_mpi.x -target nvidia-mqpu
// && mpirun -np <N> observe_mqpu_mpi.x
// ```
#include "cudaq.h"

int main() {
  cudaq::mpi::initialize();
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  double result = cudaq::observe<cudaq::parallel::mpi>(ansatz, h, 0.59);
  if (cudaq::mpi::rank() == 0)
    printf("Expectation value: %lf\n", result);
  cudaq::mpi::finalize();

  return 0;
}