// Compile and run with:
// ```
// nvq++ --target anyon anyon.cpp -o out.x && ./out.x
// ```
// This will submit the job to Anyon's default superconducting QPU.
// You can specify a specific machine via the `--anyon-machine` flag:
// ```
// nvq++ --target anyon --anyon-machine telegraph-8q anyon.cpp -o out.x &&
// ./out.x
// ```
// or for the larger system:
// ```
// nvq++ --target anyon --anyon-machine berkeley-25q anyon.cpp -o out.x &&
// ./out.x
// ```
//
// To use this target you will need to set up credentials in `~/.anyon_config`
// The configuration file should contain your Anyon Technologies username and
// password:
// ```
// credential:<username>:<password>
// ```

#include <cudaq.h>
#include <fstream>

// Define a quantum kernel to execute on Anyon backend.
struct ghz {
  // Maximally entangled state between 5 qubits.
  auto operator()() __qpu__ {
    cudaq::qvector q(5);
    h(q[0]);
    for (int i = 0; i < 4; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    auto result = mz(q);
  }
};

int main() {

  // Submit asynchronously
  auto future = cudaq::sample_async(ghz{});

  // ... classical optimization code can run while job executes ...

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

  // OR: Submit to synchronously
  auto counts = cudaq::sample(ghz{});
  counts.dump();

  return 0;
}
