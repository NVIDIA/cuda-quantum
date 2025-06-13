#include <cudaq.h>
#include <stdio.h> 

// [Begin Intrinsic Modifiers C++]
auto kernel_intrinsic_modifiers = []() __qpu__ {
  cudaq::qubit q, r, s; // Allocate 3 qubits

  // Apply T operation
  t(q);

  // Apply Tdg operation
  t<cudaq::adj>(q);

  // Apply control Hadamard operation
  // q and r are controls, s is target.
  h<cudaq::ctrl>(q, r, s);

  // Error, ctrl requires > 1 qubit operands (target + at least one control)
  // h<cudaq::ctrl>(r); // This line is commented out in the original, keep it so.

  // Add measurements to make it a complete circuit for sampling
  mz(q);
  mz(r);
  mz(s);
  printf("C++: Intrinsic modifiers kernel executed.\n");
};
// [End Intrinsic Modifiers C++]

int main() {
  // [Begin Intrinsic Modifiers C++ Execution]
  cudaq::sample(kernel_intrinsic_modifiers);
  // [End Intrinsic Modifiers C++ Execution]
  return 0;
}