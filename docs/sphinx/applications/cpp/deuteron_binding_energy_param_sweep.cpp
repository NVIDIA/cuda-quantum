#include <cudaq.h>

struct deuteron_n2_ansatz {
  void operator()(double theta) __qpu__ {
    cudaq::qarray<2> q;
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};

int main() {
  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  // Perform parameter sweep for deuteron N=2 Hamiltonian
  const auto param_space = cudaq::linspace(-M_PI, M_PI, 25);
  for (const auto &param : param_space) {
    // KERNEL::observe(...) <==>
    // E(params...) = <psi(params...) | H | psi(params...)>
    double energy_at_param = cudaq::observe(deuteron_n2_ansatz{}, h, param);
    printf("<H>(%lf) = %lf\n", param, energy_at_param);
  }
}
