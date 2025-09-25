#include <cudaq.h>

__qpu__ void kernel(int num_qubits, int num_rounds) {
    cudaq::qvector q(num_qubits);
    for (int i = 0; i < num_rounds; i++){
        h(q);
        cudaq::save_state();  
    }
    mz(q);
  }

int main() {

  int num_qubits = 5;
  int num_rounds = 3;
  double noise_bf_prob = 0.0625;

  cudaq::noise_model noise;
  cudaq::bit_flip_channel bf(noise_bf_prob);
  for (std::size_t i = 0; i < num_qubits; i++)
    noise.add_channel("mz", {i}, bf);
  cudaq::set_noise(noise);

  cudaq::ExecutionContext ctx_sample("sample", 10);
  ctx_sample.noiseModel = &noise;
  auto &platform = cudaq::get_platform();

  platform.set_exec_ctx(&ctx_sample);

  kernel(num_qubits, num_rounds);
  ctx_sample.dump_recorded_states();
  platform.reset_exec_ctx();
}

