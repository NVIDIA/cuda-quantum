// Compile and run with:
// ```
// nvq++ --target oqc oqc.cpp -o out.x && ./out.x
// ```
// This will submit the job to the OQC platform. You can also specify
// the machine to use via the `--oqc-machine` flag:
// ```
// nvq++ --target oqc --oqc-machine lucy oqc.cpp -o out.x && ./out.x
// ```
// The default is the 8 qubit Lucy device. You can set this to be either
// `toshiko` or `lucy` via this flag.
//
// To use the OQC target you will need to set the following environment
// variables: OQC_URL OQC_EMAIL OQC_PASSWORD To setup an account, contact
// oqc_qcaas_support@oxfordquantumcircuits.com

#include <cudaq.h>
#include <fstream>

// Define a simple quantum kernel to execute on OQC backends.
struct bell_state {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  }
};

int main() {
  // Submit to OQC asynchronously (e.g., continue executing
  // code in the file until the job has been returned).
  auto future = cudaq::sample_async(bell_state{});
  // ... classical code to execute in the meantime ...

  // Can write the future to file:
  {
    std::ofstream out("future.json");
    out << future;
  }

  // Then come back and read it in later.
  cudaq::async_result<cudaq::sample_result> readIn;
  std::ifstream in("future.json");
  in >> readIn;

  // Get the results of the read in future.
  auto async_counts = readIn.get();
  async_counts.dump();

  // OR: Submit to OQC synchronously (e.g., wait for the job
  // result to be returned before proceeding).
  auto counts = cudaq::sample(bell_state{});
  counts.dump();
}
