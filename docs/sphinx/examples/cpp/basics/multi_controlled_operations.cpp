// Compile and run with:
// ```
// nvq++ multi_controlled_operations.cpp -o ccnot.x && ./ccnot.x
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>

// Here we demonstrate how one might apply multi-controlled
// operations on a general CUDA Quantum kernel.
struct ApplyX {
  void operator()(cudaq::qubit &q) __qpu__ { x(q); }
};

// Generic kernel converting a gate operation into a control gate.
struct controlled_op {
  // constrain the signature of the incoming kernel
  void operator()(cudaq::takes_qubit auto &&apply_op, cudaq::qview<> control,
                  cudaq::qubit &target) __qpu__ {
    // Control U (apply_op) on the first two qubits of
    // the allocated register.
    cudaq::control(apply_op, control, target);
  }
};

// Constructing a `CCNOT` test using `controlled_op` kernel with `ApplyX` as the
// base operation.
struct ccnot_test {
  void operator()() __qpu__ {
    cudaq::qvector qs(3);
    // Bring the two control qubits into |1> state.
    x(qs);
    x(qs[1]);

    // Control U (apply_op) on the first two qubits of
    // the allocated register.
    controlled_op{}(ApplyX{}, qs.front(2), qs[2]);

    mz(qs);
  }
};

int main() {
  // We can achieve the same thing as above via
  // a lambda expression.
  auto ccnot = []() __qpu__ {
    cudaq::qvector q(3);

    x(q);
    x(q[1]);

    x<cudaq::ctrl>(q[0], q[1], q[2]);

    mz(q);
  };

  auto counts = cudaq::sample(ccnot);

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.data(), count);
  }

  auto counts2 = cudaq::sample(ccnot_test{});

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts2) {
    printf("Observed: %s, %lu\n", bits.data(), count);
  }
}
