#include <cudaq.h>
#include <stdio.h>

// [Begin Negative Polarity C++]
auto kernel_neg_polarity = []() __qpu__ {
  cudaq::qubit q, r;

  // To demonstrate the effect of !q:
  // If q is |0> (default), !q is true, X is applied to r. r becomes |1>.
  // If q is |1> (after an X(q)), !q is false, X is not applied to r. r remains
  // |0>.

  // Let's test the case where q is |1> initially.
  x(q);                  // q is now |1>
  x<cudaq::ctrl>(!q, r); // !q is false (control on |0>), so X is NOT applied to
                         // r. r remains |0>.
  mz(q, r);              // Expect bitstring "10" (q=1, r=0)

  printf("C++: Negative polarity kernel executed.\n");
};
// [End Negative Polarity C++]

int main() {
  // [Begin Negative Polarity C++ Execution]
  printf("C++ Negative Polarity Control Example:\n");
  auto counts = cudaq::sample(kernel_neg_polarity);
  counts.dump();
  // Expected output for the case where q starts as |1|:
  // { 10:[shots] }
  // [End Negative Polarity C++ Execution]
  return 0;
}
