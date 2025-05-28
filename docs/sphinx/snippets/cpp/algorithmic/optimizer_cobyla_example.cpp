#include <cudaq.h>
#include <cudaq/optimizers.h>
#include <stdio.h>

// [Begin COBYLA Example C++]
// Kernel definition
auto ansatz_cobyla = [](double theta, double phi) __qpu__ {
  cudaq::qarray<2> q;
  x(q[0]);
  ry(theta, q[1]);
  rx(phi, q[0]);
  cx(q[1], q[0]);
};

// Hamiltonian
cudaq::spin_op H_cobyla = cudaq::spin_op::z(0) * cudaq::spin_op::x(1) + 0.5 * cudaq::spin_op::y(0);

int main() {
  cudaq::optimizers::cobyla optimizer;
  optimizer.initial_parameters = std::vector<double>{0.1, 0.1};
  optimizer.max_eval = 100;

  auto [opt_val, opt_params] = optimizer.optimize(
      2, 
      [&](const std::vector<double> &params, std::vector<double> &grad_vec) {
        // grad_vec is not used by COBYLA
        return cudaq::observe(ansatz_cobyla, H_cobyla, params[0], params[1]);
      });

  printf("COBYLA Optimal value: %lf\n", opt_val);
  printf("COBYLA Optimal parameters: [%lf, %lf]\n", opt_params[0], opt_params[1]);
  return 0;
}
// [End COBYLA Example C++]