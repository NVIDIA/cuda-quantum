// Compile and run with:
// ```
// nvq++ --target quantinuum --quantinuum-machine H1-2E quantinuum.cpp -o out.x
// ./out.x
// ```
// Assumes a valid set of credentials have been stored.
// To first confirm the correctness of the program locally,
// Add a --emulate to the `nvq++` command above.

#include <cudaq.h>
#include <fstream>

// Define a simple quantum kernel to execute on Quantinuum.
struct ghz {
  // Maximally entangled state between 5 qubits.
  auto operator()() __qpu__ {
    cudaq::qreg q(5);
    h(q[0]);
    for (int i = 0; i < 4; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  // Submit to Quantinuum asynchronously. E.g, continue executing
  // code in the file until the job has been returned.
  auto future = cudaq::sample_async(ghz{});
  // ... classical code to execute in the meantime ...

  // Can write the future to file:
  {
    std::ofstream out("saveMe.json");
    out << future;
  }

  // Then come back and read it in later.
  cudaq::async_result<cudaq::sample_result> readIn;
  std::ifstream in("saveMe.json");
  in >> readIn;

  // Get the results of the read in future.
  auto async_counts = readIn.get();
  async_counts.dump();

  // OR: Submit to Quantinuum synchronously. E.g, wait for the job
  // result to be returned before proceeding.
  auto counts = cudaq::sample(ghz{});
  counts.dump();
}
