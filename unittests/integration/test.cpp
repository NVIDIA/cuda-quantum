#include <cudaq.h>

__qpu__ auto kernel(int num_qubits, int num_rounds) {
    cudaq::qvector q(num_qubits);
    // cudaq::save_state();  
    for (int i = 0; i < num_qubits; i++) {
      cudaq::apply_noise<cudaq::pauli1>(0.1, 0.1, 0.1, q[i]);
    }
    for (int i = 0; i < num_rounds; i++){
        h(q[1]);
        cudaq::save_state();  
        x<cudaq::ctrl>(q[1], q[0]);
        cudaq::save_state(); 
    }
  }

int main() {

  int num_qubits = 2;
  int num_rounds = 1;

  /*
  cudaq::ExecutionContext ctx_sample("sample", 1);
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_sample);

  kernel(num_qubits, num_rounds);

  std::cout << "After kernel\n";
  std::cout << "--------------------------------\n";
  ctx_sample.dump_recorded_states(); 
  */

  double noise_bf_prob = 1.;

  cudaq::noise_model noise;
  cudaq::depolarization_channel depolarization(noise_bf_prob);

  noise.add_channel<cudaq::types::h>({0}, depolarization);
  noise.add_channel<cudaq::types::h>({1}, depolarization);
  noise.add_channel<cudaq::types::x>({0}, depolarization);


  cudaq::set_noise(noise);

  cudaq::ExecutionContext ctx_msm_size("training", 1);
  auto &platform2 = cudaq::get_platform();
  ctx_msm_size.set_seed(42);

  platform2.set_exec_ctx(&ctx_msm_size);

  kernel(num_qubits, num_rounds);

  ctx_msm_size.dump_recorded_states();
  platform2.reset_exec_ctx();
}

