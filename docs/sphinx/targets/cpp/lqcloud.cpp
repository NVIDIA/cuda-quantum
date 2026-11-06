// Set LQCLOUD_API_KEY in the environment, then compile with:
// ```
// nvq++ --target lqcloud \
//   --lqcloud-url "${LQCLOUD_URL:-https://cloud.logicalqubit.com}" \
//   --lqcloud-qpu "${LQCLOUD_QPU:-MQ02}" \
//   lqcloud.cpp -o lqcloud.x
// ```
// Select an available QPU by its exact name in https://cloud.logicalqubit.com.
// MQ02 is an example; LQCLOUD_URL and LQCLOUD_QPU override the defaults above.
// The credential is read from the environment and must not be placed on the
// command line.

#include <cudaq.h>

struct bell {
  void operator()() __qpu__ {
    cudaq::qvector qubits(2);
    h(qubits[0]);
    x<cudaq::ctrl>(qubits[0], qubits[1]);
    mz(qubits);
  }
};

int main() {
  constexpr std::size_t shots = 5;
  bell circuit;

  auto counts = cudaq::sample(shots, circuit);
  counts.dump();

  // Consume the asynchronous result in this same process.
  auto future = cudaq::sample_async(shots, 0, circuit);
  future.get().dump();
}
