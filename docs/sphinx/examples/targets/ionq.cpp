// Compile and run with:
// ```
// nvq++ --target ionq ionq.cpp -o out.x && ./out.x
// ```
// Assumes a valid set of credentials have been stored. 

// Define a simple quantum kernel to execute on IonQ.
struct ghz {
  // Maximally entangled state between 5 qubits.
  auto operator()() __qpu__ {
    cudaq::qreg q(5);
    h(q[0]);
    for (int i = 0; i < 5 - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  // Submit to IonQ synchronously. E.g, wait for the job
  // result to be returned before proceeding. 
  auto counts = cudaq::sample(ghz{});
  counts.dump();

  // Submit to IonQ asynchronously. E.g, continue executing
  // code in the file until the job has been returned.
  auto future = cudaq::sample_async(ghz{});
  // ... classical code to execute in the meantime ...
  auto async_counts = future.get();
  async_counts.dump();
}
