#include <cudaq.h>
#include <cudaq/noise_model.h>
#include <iostream>
#include <vector>
#include <complex>

// Define a simple bit-flip channel for demonstration
// K0 = sqrt(1-p) * I, K1 = sqrt(p) * X
// p = 0.1 (10% bit-flip probability)
// K0 = [[sqrt(0.9), 0], [0, sqrt(0.9)]]
// K1 = [[0, sqrt(0.1)], [sqrt(0.1), 0]]
// Flattened row-major:
// K0_flat = {sqrt(0.9), 0, 0, sqrt(0.9)}
// K1_flat = {0, sqrt(0.1), sqrt(0.1), 0}
cudaq::kraus_channel create_bit_flip_channel(double p) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("Probability p must be between 0 and 1.");
    }
    using namespace std::complex_literals;
    double sqrt_1_minus_p = std::sqrt(1.0 - p);
    double sqrt_p = std::sqrt(p);

    cudaq::kraus_op k0_op({sqrt_1_minus_p + 0.0i, 0.0i, 0.0i, sqrt_1_minus_p + 0.0i});
    cudaq::kraus_op k1_op({0.0i, sqrt_p + 0.0i, sqrt_p + 0.0i, 0.0i});
    return cudaq::kraus_channel({k0_op, k1_op});
}

// Kernel to test noise models
struct ghz_cpp {
  void operator()() __qpu__ {
    cudaq::qarray<3> q;
    h(q[0]);
    cx(q[0], q[1]);
    cx(q[1], q[2]);
    // Apply some gates that will have noise
    z(q[0]); // Noise on specific qubit
    x(q[1]); // Noise on all qubits
    rx(1.23, q[2]); // Dynamic noise
    mz(q);
  }
};

int main() {
  auto bit_flip_channel = create_bit_flip_channel(0.1); // 10% bit flip

  cudaq::noise_model noise; // Renamed from 'noise' in RST to avoid conflict if global

  // [Begin CPP AddChannelSpecific]
  // Add a noise channel to z gate on qubit 0
  noise.add_channel("z", {0}, bit_flip_channel);
  // [End CPP AddChannelSpecific]

  // [Begin CPP AddChannelAllQubit]
  // Add a noise channel to x gate, regardless of qubit operands.
  noise.add_all_qubit_channel("x", bit_flip_channel);
  // [End CPP AddChannelAllQubit]

  // [Begin CPP AddChannelDynamic]
  // Add a dynamic noise channel to the 'rx' gate.
  noise.add_channel("rx",
      [&](const std::vector<std::size_t> &qubits, const std::vector<double> &params) -> cudaq::kraus_channel {
          std::cout << "Dynamic noise callback for rx on qubits: ";
          for(const auto& q_idx : qubits) std::cout << q_idx << " ";
          std::cout << "with params: ";
          for(const auto& p_val : params) std::cout << p_val << " ";
          std::cout << std::endl;
          // For simplicity, return the same bit-flip channel,
          // but could be dependent on qubits/params.
          // Example: higher error for larger rotation angles
          double p_dynamic = 0.05; // Default dynamic probability
          if (!params.empty() && params[0] > M_PI_2) { // if angle > pi/2
              p_dynamic = 0.15; // higher error
          }
          return create_bit_flip_channel(p_dynamic);
      });
  // [End CPP AddChannelDynamic]

  cudaq::set_noise(noise);
  std::cout << "C++: Running GHZ with noise model." << std::endl;
  auto counts = cudaq::sample(ghz_cpp{});
  cudaq::unset_noise();

  std::cout << "C++: Counts with noise:" << std::endl;
  counts.dump();

  // For comparison, run without noise
  std::cout << "\nC++: Running GHZ without noise model." << std::endl;
  auto ideal_counts = cudaq::sample(ghz_cpp{});
  std::cout << "C++: Ideal counts:" << std::endl;
  ideal_counts.dump();

  return 0;
}