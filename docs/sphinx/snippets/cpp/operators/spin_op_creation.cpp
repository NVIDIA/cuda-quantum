#include <cudaq.h>
#include <cudaq/spin_op.h>
#include <iostream> // For std::cout and std::cerr

// [Begin SpinOp Creation C++]
// The following lines define the spin_op `h` as shown in the documentation.
auto h_cpp_example =
    5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
    2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
    .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
// [End SpinOp Creation C++]

int main() {
  // [Begin SpinOp Creation C++ Execution]
  // This part is for testing and making the snippet runnable.
  // It won't be included in the RST by the literalinclude directive.
  std::cout << "C++ Spin Operator h_cpp_example:\n"
            << h_cpp_example.to_string() << std::endl;

  if (h_cpp_example.get_term_count() == 0) {
    std::cerr << "Error: C++ Spin operator h_cpp_example has no terms."
              << std::endl;
    return 1; // Indicate failure
  }

  // Example of using the spin_op in an observe call with a simple ansatz.
  // The spin_op uses qubits 0 and 1, so we need at least 2 qubits.
  struct ansatz_cpp {
    void operator()() __qpu__ {
      cudaq::qarray<2> q;
      x(q[0]); // A simple operation on one of the qubits.
    }
  };

  double energy = cudaq::observe(ansatz_cpp{}, h_cpp_example);
  std::cout << "Observed energy (C++): " << energy << std::endl;
  // [End SpinOp Creation C++ Execution]
  return 0; // Indicate success
}
