// Compile and run with:
// ```
// nvq++ bernstein_vazirani.cpp -o bv.x --target nvidia-mgpu
// mpirun -np 4 ./bv.x
// ```

// This example is meant to demonstrate the cuQuantum GPU-accelerated multi-GPU
// backend. A state vector simulation for up to 32 qubits can be run on a single
// GPU (if the GPU memory is large enough). Larger state vectors can be
// simulated using the multi-GPU backend. The amount of resources required for
// the simulation doubles with with each additional qubit.

#include <bitset>
#include <cudaq.h>
#include <iostream>
#include <random>

#ifndef SIZE
#define SIZE 5
#endif

template <int nrOfBits>
std::bitset<nrOfBits> random_bits(int seed) {

  std::bitset<nrOfBits> randomBits;
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  for (size_t i = 0; i < nrOfBits; i++) {
    randomBits.set(i, distribution(generator) < 0.5 ? 0 : 1);
  }
  return randomBits;
}

template <int nrOfBits>
struct oracle {
  auto operator()(std::bitset<nrOfBits> bitvector, cudaq::qview<> qs,
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
  auto operator()(std::bitset<nrOfBits> bitvector) __qpu__ {

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

int main(int argc, char *argv[]) {
  auto seed = 1 < argc ? atoi(argv[1]) : time(NULL);

  // The number of qubits can be >32 when targeting the `nvidia-mgpu` backend.
  const int nr_qubits = SIZE;
  const int nr_shots = 100;
  auto bitvector = random_bits<nr_qubits>(seed);
  auto kernel = bernstein_vazirani<nr_qubits>{};
  auto counts = cudaq::sample(nr_shots, kernel, bitvector);

  if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
    printf("Encoded bitstring:  %s\n", bitvector.to_string().c_str());
    printf("Measured bitstring: %s\n\n", counts.most_probable().c_str());

    for (auto &[bits, count] : counts) {
      printf("observed %s with %.0f%% probability\n", bits.data(),
             100.0 * count / nr_shots);
    }
  }

  return 0;
}
