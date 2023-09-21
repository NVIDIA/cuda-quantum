// Compile and run with:
// ```
// nvq++ --target iqm iqm.cpp --iqm-machine Adonis -o out.x && ./out.x
// ```
// Assumes a valid set of credentials have been stored.

#include <cudaq.h>
#include <fstream>

// Define a simple quantum kernel to execute on IQM Server.
struct adonis_ghz {
  // Maximally entangled state between 5 qubits on Adonis QPU.
  //       QB1
  //        |
  // QB2 - QB3 - QB4
  //        |
  //       QB5

  auto operator()() __qpu__ {
    cudaq::qreg q(5);
    h(q[2]); // QB3

    x<cudaq::ctrl>(q[2], q[0]);
    x<cudaq::ctrl>(q[2], q[1]);
    x<cudaq::ctrl>(q[2], q[3]);
    x<cudaq::ctrl>(q[2], q[4]);

    mz(q);
  }
};

int main() {
  // Submit to IQM Server asynchronously. E.g, continue executing
  // code in the file until the job has been returned.
  auto future = cudaq::sample_async(adonis_ghz{});
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

  // OR: Submit to IQM Server synchronously. E.g, wait for the job
  // result to be returned before proceeding.
  auto counts = cudaq::sample(adonis_ghz{});
  counts.dump();
}
