// Compile and run with:
// ```
// nvq++ --target nvqc --nvqc-nqpus 3 nvqc_vqe.cpp -o out.x
// ./out.x
// ```
// Note: we set `nqpus` to 3 to establish 3 concurrent NVQC job submission
// pipes.
// Assumes a valid NVQC API key has been set in the `NVQC_API_KEY` environment
// variable. Please refer to the documentations for information about how to
// attain NVQC API key.

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>
#include <iostream>

int main() {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto [ansatz, theta] = cudaq::make_kernel<double>();
  auto q = ansatz.qalloc();
  auto r = ansatz.qalloc();
  ansatz.x(q);
  ansatz.ry(theta, r);
  ansatz.x<cudaq::ctrl>(r, q);

  // Run VQE with a gradient-based optimizer.
  // Delegate cost function and gradient computation across different NVQC-based
  // QPUs. Note: depending on the user's account, there might be different
  // number of NVQC worker instances available. Hence, although we're making
  // concurrent job submissions across multiple QPUs, the speedup would be
  // determined by the number of NVQC worker instances.
  cudaq::optimizers::lbfgs optimizer;
  auto [opt_val, opt_params] = optimizer.optimize(
      /*dim=*/1, /*opt_function*/ [&](const std::vector<double> &params,
                                      std::vector<double> &grads) {
        // Queue asynchronous jobs to do energy evaluations across multiple QPUs
        auto energy_future =
            cudaq::observe_async(/*qpu_id=*/0, ansatz, h, params[0]);
        const double paramShift = M_PI_2;
        auto plus_future = cudaq::observe_async(/*qpu_id=*/1, ansatz, h,
                                                params[0] + paramShift);
        auto minus_future = cudaq::observe_async(/*qpu_id=*/2, ansatz, h,
                                                 params[0] - paramShift);
        grads[0] = (plus_future.get().expectation() -
                    minus_future.get().expectation()) /
                   2.0;
        return energy_future.get().expectation();
      });
  std::cout << "Minimum energy = " << opt_val << " (expected -1.74886).\n";
}
