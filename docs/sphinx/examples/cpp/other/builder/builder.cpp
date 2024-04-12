/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ builder.cpp -o builder.x && ./builder.x
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/builder.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>

static bool results_are_close(const cudaq::sample_result &f1,
                              const cudaq::sample_result &f2) {
  // Stub for a fancy comparison.
  f1.dump();
  f2.dump();
  return true;
}

// This example demonstrates various uses for the `cudaq::builder`
// type. This type enables one to dynamically construct callable
// CUDA Quantum kernels via just-in-time compilation. The typical workflow
// starts by creating a `cudaq::builder` and any CUDA Quantum kernel runtime
// arguments via the `cudaq::make_kernel<ParameterTypes...>()` function.
// Programmers get reference to the builder and the concrete runtime
// parameters for the kernel function, and can then begin building up
// the CUDA Quantum kernel. Once done adding gates, the builder itself is
// callable, and can be used just like any other CUDA Quantum kernel.

int main() {
  {

    // Create a Hamiltonian as a `cudaq::spin_op`.
    using namespace cudaq::spin;
    cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                       .21829 * z(0) - 6.125 * z(1);

    // Build a quantum kernel dynamically
    // Start by creating the Builder, the kernel argument types
    // should be provided here as template parameters.
    auto [ansatz, theta] = cudaq::make_kernel<double>();

    // Allocate some qubits
    auto q = ansatz.qalloc(2);

    // Build up the circuit, use the acquired parameter
    ansatz.x(q[0]);
    ansatz.ry(theta, q[1]);
    ansatz.x<cudaq::ctrl>(q[1], q[0]); // Need to get rid of ::

    // The buildable kernel can be passed to CUDA Quantum functions
    // just like a declared kernel type.
    ansatz(.59);
    double exp = cudaq::observe(ansatz, h, .59);
    printf("<H2> = %lf\n", exp);
  }

  {
    // Build up a 2 parameter circuit using a vector<double> parameter
    // Run the CUDA Quantum optimizer to find optimal value.
    using namespace cudaq::spin;
    cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                       .21829 * z(0) - 6.125 * z(1);
    cudaq::spin_op h3 = h + 9.625 - 9.625 * z(2) - 3.913119 * x(1) * x(2) -
                        3.913119 * y(1) * y(2);

    // Create the kernel with signature void(std::vector<double>)
    auto [ansatz, thetas] = cudaq::make_kernel<std::vector<double>>();

    // Allocate some qubits
    auto q = ansatz.qalloc(3);

    // Build the kernel
    ansatz.x(q[0]);
    ansatz.ry(thetas[0], q[1]);
    ansatz.ry(thetas[1], q[2]);
    ansatz.x<cudaq::ctrl>(q[2], q[0]);
    ansatz.x<cudaq::ctrl>(q[0], q[1]);
    // Can do fancy arithmetic with Parameter types.
    ansatz.ry(-thetas[0], q[1]);
    // -or- `ansatz_builder.ry(-1.0 * thetas[0], q[1]);`
    // -or- `ansatz_builder.ry(thetas[0] * -1.0, q[1]);`
    // -or- `ansatz_builder.ry(-1 * thetas[0], q[1]);`
    ansatz.x<cudaq::ctrl>(q[0], q[1]);
    ansatz.x<cudaq::ctrl>(q[1], q[0]);

    // You now have a callable kernel, use it with the VQE algorithm
    cudaq::gradients::central_difference gradient(ansatz);
    cudaq::optimizers::lbfgs optimizer;
    auto [opt_val_0, optpp] = cudaq::vqe(ansatz, gradient, h3, optimizer, 2);
    printf("<H3> = %lf\n", opt_val_0);
  }

  {
    // Make a kernel for sampling, here the GHZ state on 8 qubits
    int n_qubits = 8;
    auto ghz_builder = cudaq::make_kernel();

    // Allocate the qubits
    auto q = ghz_builder.qalloc(n_qubits);

    // Build the GHZ state
    ghz_builder.h(q[0]);
    for (int i = 0; i < n_qubits - 1; i++) {
      ghz_builder.x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    ghz_builder.mz(q);

    // You can get the MLIR representation
    auto mlir_code = ghz_builder.to_quake();

    // Sample and get the counts
    auto counts = cudaq::sample(ghz_builder);
    counts.dump();
  }

  {
    // In a simulated environment, it is sometimes useful to be able to
    // specify an initial state vector. The initial state vector is 2 to the
    // power `n` where `n` is the number of qubits.

    // In this example, we create a kernel template `sim_kernel` that captures
    // the variable `init_state` by reference.
    auto sim_builder = cudaq::make_kernel();
    std::vector<cudaq::simulation_scalar> init_state;
    auto q = sim_builder.qalloc(init_state);
    // Build the quantum circuit template here.
    sim_builder.mz(q);

    // Now we are ready to instantiate the kernel and invoke it. So we can set
    // the `init_state` to a vector with 2 complex values (1 qubit) and
    // get the results.
    init_state = {{0.0, 1.0}, {1.0, 0.0}};
    auto counts0 = cudaq::sample(sim_builder);

    // Now suppose we have a different initial state with 4 complex values (2
    // qubits). Let's rerun the kernel with the new `init_state`.
    init_state = {{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}};
    auto counts1 = cudaq::sample(sim_builder);

    // Finally in this wholly contrived example, we test the results to make
    // sure they are "close".
    if (results_are_close(counts0, counts1)) {
      printf("The two initial states generated results that are \"close\".\n");
    }
  }

  {

    // Let's do a final sampling task. Let's
    // build up an element of the Toffoli truth table.
    auto ccnot_builder = cudaq::make_kernel();
    auto q = ccnot_builder.qalloc(3);
    ccnot_builder.x(q);
    ccnot_builder.x(q[1]);
    ccnot_builder.x<cudaq::ctrl>(q[0], q[1], q[2]);
    ccnot_builder.mz(q);

    auto mlir_code = ccnot_builder.to_quake();
    printf("%s\n", mlir_code.c_str());

    auto counts = cudaq::sample(ccnot_builder);
    counts.dump();
  }

  return 0;
}
