// Compile and run with:
// ```
// nvq++ noise_modeling.cpp -o noise.x && ./noise.x
// ```

#include "cudaq.h"

int main() {
  // Define a  kernel
  auto xgate = []() __qpu__ {
    cudaq::qubit q;
    x(q);
    mz(q);
  };

  // Run noise-less simulation
  auto counts = cudaq::sample(xgate);
  counts.dump();

  // Create a depolarizing Kraus channel made up of two Kraus operators.
  cudaq::kraus_channel depol({cudaq::complex{0.99498743710662, 0.0},
                              {0.0, 0.0},
                              {0.0, 0.0},
                              {0.99498743710662, 0.0}},

                             {cudaq::complex{0.0, 0.0},
                              {0.05773502691896258, 0.0},
                              {0.05773502691896258, 0.0},
                              {0.0, 0.0}},

                             {cudaq::complex{0.0, 0.0},
                              {0.0, -0.05773502691896258},
                              {0.0, 0.05773502691896258},
                              {0.0, 0.0}},

                             {cudaq::complex{0.05773502691896258, 0.0},
                              {0.0, 0.0},
                              {0.0, 0.0},
                              {-0.05773502691896258, 0.0}});

  // Create the noise model
  cudaq::noise_model noise;
  // Add the Kraus channel to the x operation on qubit 0.
  noise.add_channel<cudaq::types::x>({0}, depol);

  // Set the noise model
  cudaq::set_noise(noise);

  // Run the noisy simulation
  counts = cudaq::sample(xgate);
  counts.dump();

  // Unset the noise model when done. This is not necessary in this case but is
  // good practice in order to not interfere with future simulations.
  cudaq::unset_noise();
}
