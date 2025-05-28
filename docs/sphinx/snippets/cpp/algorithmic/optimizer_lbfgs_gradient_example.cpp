#include <cudaq.h>
#include <cudaq/optimizers.h>
#include <cudaq/gradients.h>
#include <stdio.h>
#include <tuple> // Required for std::make_tuple

// [Begin LBFGS Example C++]
auto deuteron_n3_ansatz = [](double x0, double x1) __qpu__ {
  cudaq::qarray<3> q;
  x(q[0]);
  ry(x0, q[1]);
  ry(x1, q[2]);
  x<cudaq::ctrl>(q[2], q[0]);
  x<cudaq::ctrl>(q[0], q[1]); // Assuming vctrl was a typo for ctrl
  ry(-x0, q[1]);
  x<cudaq::ctrl>(q[0], q[1]);
  x<cudaq::ctrl>(q[1], q[0]);
};

cudaq::spin_op h_deut_lbfgs = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
                      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
                      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
cudaq::spin_op h3_deut_lbfgs = h_deut_lbfgs + 9.625 - 9.625 * cudaq::spin_op::z(2) -
                       3.913119 * cudaq::spin_op::x(1) * cudaq::spin_op::x(2) -
                       3.913119 * cudaq::spin_op::y(1) * cudaq::spin_op::y(2);

auto argsMapper_lbfgs = [](const std::vector<double>& x_params) {
  return std::make_tuple(x_params[0], x_params[1]);
};

int main() {
  cudaq::gradients::central_difference gradient(deuteron_n3_ansatz, argsMapper_lbfgs);
  cudaq::optimizers::lbfgs optimizer;
  optimizer.initial_parameters = std::vector<double>{0.5, 0.5};

  auto [min_val, opt_params] = optimizer.optimize(
      2, 
      [&](const std::vector<double>& x_lambda, std::vector<double>& grad_vec) {
        auto cost = cudaq::observe(deuteron_n3_ansatz, h3_deut_lbfgs, x_lambda[0], x_lambda[1]);
        grad_vec = gradient.compute(x_lambda, [&](const std::vector<double>& p_grad) {
            return cudaq::observe(deuteron_n3_ansatz, h3_deut_lbfgs, p_grad[0], p_grad[1]);
        }, cost);
        return cost;
      });

  printf("L-BFGS Optimizer found %lf at [%lf,%lf]\n", min_val, opt_params[0], opt_params[1]);
  return 0;
}
// [End LBFGS Example C++]