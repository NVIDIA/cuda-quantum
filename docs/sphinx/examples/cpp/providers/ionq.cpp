// Compile and run with:
// ```
// nvq++ --target ionq ionq.cpp -o out.x && ./out.x
// ```
// This will submit the job to the IonQ ideal simulator target (default).
// Alternatively, we can enable hardware noise model simulation by specifying
// the `--ionq-noise-model`, e.g.,
// ```
// nvq++ --target ionq --ionq-machine simulator --ionq-noise-model aria-1
// ionq.cpp -o out.x && ./out.x
// ```
// where we set the noise model to mimic the 'aria-1' hardware device.
// Please refer to your IonQ Cloud dashboard for the list of simulator noise
// models.
// Note: `--ionq-machine simulator` is  optional since 'simulator' is the
// default configuration if not provided. Assumes a valid set of credentials
// have been stored.

#include <cudaq.h>
#include <fstream>

// Define a simple quantum kernel to execute on IonQ.
struct ghz {
  // Maximally entangled state between 5 qubits.
  auto operator()() __qpu__ {
    cudaq::qreg q(5);
    h(q[0]);
    for (int i = 0; i < 4; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    auto result = mz(q);
  }
};

int main() {
  // Submit to IonQ asynchronously. E.g, continue executing
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

  // OR: Submit to IonQ synchronously. E.g, wait for the job
  // result to be returned before proceeding.
  auto counts = cudaq::sample(ghz{});
  counts.dump();
}
