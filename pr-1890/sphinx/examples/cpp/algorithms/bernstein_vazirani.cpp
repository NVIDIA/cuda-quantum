// Compile and run with:
// ```
// nvq++ bernstein_vazirani.cpp -o bv.x --target nvidia --target-option mgpu
// mpirun -np 4 ./bv.x
// ```

// This example is meant to demonstrate the cuQuantum GPU-accelerated multi-GPU
// backend. A state vector simulation for up to 32 qubits can be run on a single
// GPU (if the GPU memory is large enough). Larger state vectors can be
// simulated using the multi-GPU backend. The amount of resources required for
// the simulation doubles with with each additional qubit.

#include <cudaq.h>
#include <iostream>
#include <random>
#include <vector>

#ifndef SIZE
#define SIZE 5
#endif

template <int nrOfBits>
std::vector<bool> random_bits(int seed) {

  std::vector<bool> randomBits(nrOfBits);
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  for (size_t i = 0; i < nrOfBits; i++) {
    randomBits[i] = distribution(generator) >= 0.5;
  }
  return randomBits;
}

template <int nrOfBits>
struct oracle {
  auto operator()(std::vector<bool> bitvector, cudaq::qview<> qs,
                  cudaq::qubit &aux) __qpu__ {

    for (size_t i = 0; i < nrOfBits; i++) {
      if (bitvector[i] & 1) {
        x<cudaq::ctrl>(qs[nrOfBits - i - 1], aux);
      }
    }
  }
};

template <int nrOfBits>
struct bernstein_vazirani {
  auto operator()(std::vector<bool> bitvector) __qpu__ {

    cudaq::qarray<nrOfBits> qs;
    cudaq::qubit aux;
    h(aux);
    z(aux);
    h(qs);

    oracle<nrOfBits>{}(bitvector, qs, aux);
    h(qs);
    mz(qs);
  }
};

// Construct the bit vector such that the last bit has highest significance.
std::string asString(const std::vector<bool> &bitvector) {
  char *buffer = static_cast<char *>(alloca(bitvector.size() + 1));
  std::size_t N = bitvector.size();
  buffer[N] = '\0';
  for (std::size_t i = 0; i < N; ++i)
    buffer[N - 1 - i] = '0' + bitvector[i];
  return {buffer, N};
}

int main(int argc, char *argv[]) {
  auto seed = 1 < argc ? atoi(argv[1]) : time(NULL);

  // The number of qubits can be >32 when targeting the `nvidia-mgpu` backend.
  const int nr_qubits = SIZE;
  const int nr_shots = 100;
  auto bitvector = random_bits<nr_qubits>(seed);
  auto kernel = bernstein_vazirani<nr_qubits>{};
  auto counts = cudaq::sample(nr_shots, kernel, bitvector);

  if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
    printf("Encoded bitstring:  %s\n", asString(bitvector).c_str());
    printf("Measured bitstring: %s\n\n", counts.most_probable().c_str());

    for (auto &[bits, count] : counts) {
      printf("observed %s with %.0f%% probability\n", bits.data(),
             100.0 * count / nr_shots);
    }
  }

  return 0;
}
