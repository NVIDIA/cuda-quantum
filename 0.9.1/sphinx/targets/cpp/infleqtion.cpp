// Compile and run with:
// ```
// nvq++ --target infleqtion infleqtion.cpp -o out.x && ./out.x
// ```
// This will submit the job to the Infleqtion's ideal simulator,
// cq_sqale_simulator (default). Alternatively, we can enable hardware noise
// model simulation by specifying `noise-sim` to the flag `--infleqtion-method`,
// e.g.,
// ```
// nvq++ --target infleqtion --infleqtion-machine cq_sqale_qpu
// --infleqtion-method noise-sim infleqtion.cpp -o out.x && ./out.x
// ```
// where "noise-sim" instructs Superstaq to perform a noisy emulation of the
// QPU. An ideal dry-run execution on the QPU may be performed by passing
// `dry-run` to the `--infleqtion-method` flag, e.g.,
// ```
// nvq++ --target infleqtion --infleqtion-machine cq_sqale_qpu
// --infleqtion-method dry-run infleqtion.cpp -o out.x && ./out.x
// ```
// Note: If targeting ideal cloud simulation, `--infleqtion-machine
// cq_sqale_simulator` is optional since it is the default configuration if not
// provided.

#include <cudaq.h>
#include <fstream>

// Define a simple quantum kernel to execute on Infleqtion backends.
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
  // Submit to infleqtion asynchronously (e.g., continue executing
  // code in the file until the job has been returned).
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

  // OR: Submit to infleqtion synchronously (e.g., wait for the job
  // result to be returned before proceeding).
  auto counts = cudaq::sample(ghz{});
  counts.dump();
}
