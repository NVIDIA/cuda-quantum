#include <cudaq.h>

/*

std::vector<std::vector<uint8_t>>  detection_matrix(
     operation statePrep,
    bool run_mz_circuit, bool keep_x_stabilizers, bool keep_z_stabilizers) {

  if (!code.contains_operation(statePrep))
    throw std::runtime_error("prep kernel not found.");

  if (!keep_x_stabilizers && !keep_z_stabilizers)
    throw std::runtime_error(" no stabilizers to keep.");

  std::vector<std::vector<uint8_t>> detection_matrix; // to return

  std::size_t numCols = numAncx + numAncz;

  cudaq::ExecutionContext ctx_msm_size("msm_size");
  ctx_msm_size.noiseModel = &noise;
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_msm_size);

  // Run the memory circuit experiment
  if (run_mz_circuit) {
    memory_circuit_mz(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  } else {
    memory_circuit_mx(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  }

  platform.reset_exec_ctx();

  if (!ctx_msm_size.msm_dimensions.has_value()) {
    throw std::runtime_error("dem_from_memory_circuit error: no MSM dimensions "
                             "found. One reason could be missing a target.");
  }
  if (ctx_msm_size.msm_dimensions.value().second == 0) {
    throw std::runtime_error(
        "dem_from_memory_circuit error: no noise mechanisms found in circuit. "
        "Cannot generate a DEM. Did you forget to enable noise?");
  }

  cudaq::ExecutionContext ctx_msm("msm");
  ctx_msm.noiseModel = &noise;
  ctx_msm.msm_dimensions = ctx_msm_size.msm_dimensions;
  platform.set_exec_ctx(&ctx_msm);

  // Run the memory circuit experiment
  if (run_mz_circuit) {
    memory_circuit_mz(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  } else {
    memory_circuit_mx(stabRound, prep, numData, numAncx, numAncz, numRounds,
                      xVec, zVec);
  }

  platform.reset_exec_ctx();

  // Populate error rates and error IDs
  dem.error_rates = std::move(ctx_msm.msm_probabilities.value());
  dem.error_ids = std::move(ctx_msm.msm_prob_err_id.value());

  auto msm_as_strings = ctx_msm.result.sequential_data();
  cudaqx::tensor<uint8_t> msm_data(
      std::vector<std::size_t>({ctx_msm_size.msm_dimensions->first,
                                ctx_msm_size.msm_dimensions->second}));
  cudaqx::tensor<uint8_t> mzTable(msm_as_strings);
  mzTable = mzTable.transpose();
  std::size_t numNoiseMechs = mzTable.shape()[1];

  std::size_t numSyndromesPerRound = numXStabs + numZStabs;

  // Populate dem.detector_error_matrix by XORing consecutive rounds. Generally
  // speaking, this is calculating H = D*Ω, where H is the Detector Error
  // Matrix, D is the Detector Matrix, and Ω is Measurement Syndrome Matrix.
  // However, D is very sparse, and is it represents simple XORs of a syndrome
  // with the prior round's syndrome.
  // Reference: https://arxiv.org/pdf/2407.13826

  auto numReturnSynPerRound = numSyndromesPerRound;

  if (keep_x_stabilizers && !keep_z_stabilizers) {
    numReturnSynPerRound = numXStabs;
  } else if (!keep_x_stabilizers && keep_z_stabilizers) {
    numReturnSynPerRound = numZStabs;
  }

  // If we are returning only x-stabilizers, we need to offset the syndrome
  // indices of mzTable by numSyndromesPerRound / 2.
  auto offset = keep_x_stabilizers && !keep_z_stabilizers ? numZStabs : 0;
  dem.detector_error_matrix = cudaqx::tensor<uint8_t>(
      {numRounds * numReturnSynPerRound, numNoiseMechs});
  for (std::size_t round = 0; round < numRounds; round++) {
    if (round == 0) {
      for (std::size_t syndrome = 0; syndrome < numReturnSynPerRound;
           syndrome++) {
        for (std::size_t noise_mech = 0; noise_mech < numNoiseMechs;
             noise_mech++) {
          dem.detector_error_matrix.at(
              {round * numReturnSynPerRound + syndrome, noise_mech}) =
              mzTable.at({round * numSyndromesPerRound + syndrome + offset,
                          noise_mech});
        }
      }
    } else {
      for (std::size_t syndrome = 0; syndrome < numReturnSynPerRound;
           syndrome++) {
        for (std::size_t noise_mech = 0; noise_mech < numNoiseMechs;
             noise_mech++) {
          dem.detector_error_matrix.at(
              {round * numReturnSynPerRound + syndrome, noise_mech}) =
              mzTable.at({round * numSyndromesPerRound + syndrome + offset,
                          noise_mech}) ^
              mzTable.at(
                  {(round - 1) * numSyndromesPerRound + syndrome + offset,
                   noise_mech});
        }
      }
    }
  }

  // Uncomment for debugging:
  // printf("dem.detector_error_matrix:\n");
  // dem.detector_error_matrix.dump_bits();

  // Populate dem.observables_flips_matrix by converting the physical data qubit
  // measurements to logical observables.
  auto first_data_row = numRounds * numSyndromesPerRound;
  assert(first_data_row < mzTable.shape()[0]);

  cudaqx::tensor<uint8_t> msm_obs(
      {mzTable.shape()[0] - first_data_row, numNoiseMechs});
  for (std::size_t row = first_data_row; row < mzTable.shape()[0]; row++)
    for (std::size_t col = 0; col < numNoiseMechs; col++)
      msm_obs.at({row - first_data_row, col}) = mzTable.at({row, col});

  // Populate dem.observables_flips_matrix by converting the physical data qubit
  // measurements to logical observables.
  dem.observables_flips_matrix = obs_matrix.dot(msm_obs) % 2;

  // printf("getting obs_matrix : \n");
  // obs_matrix.dump_bits();

  // printf("getting msm_obs : \n");
  // msm_obs.dump_bits();

  // Uncomment print statements for debugging:
  // printf("dem.detector_error_matrix Before canonicalization:\n");
  // dem.detector_error_matrix.dump_bits();
  // printf("dem.observables_flips_matrix Before canonicalization:\n");
  // dem.observables_flips_matrix.dump_bits();
  dem.canonicalize_for_rounds(numReturnSynPerRound);
  // printf("dem.detector_error_matrix After canonicalization:\n");
  // dem.detector_error_matrix.dump_bits();
  // printf("dem.observables_flips_matrix After canonicalization:\n");
  // dem.observables_flips_matrix.dump_bits();

  return dem;
}
*/

__qpu__ int sx(cudaq::qview<> qubits, cudaq::qview<> ancillas) {}

__qpu__ int sz(cudaq::qview<> qubits, cudaq::qview<> ancillas) {}

__qpu__ auto kernel(int num_qubits, int num_rounds) {
  cudaq::qvector q(num_qubits);

  for (int i = 0; i < num_qubits; i++) {
    cudaq::apply_noise<cudaq::pauli1>(0.1, 0.1, 0.1, q[i]);
  }

  for (int i = 0; i < num_rounds; i++) {
    h(q[1]);
    cudaq::save_state();
    x<cudaq::ctrl>(q[1], q[0]);
    cudaq::save_state();
  }
  return cudaq::to_integer(mz(q));
}

int main() {

  int num_qubits = 2;
  int num_rounds = 1;

  double noise_bf_prob = 1.;

  cudaq::noise_model noise;
  cudaq::depolarization_channel depolarization(noise_bf_prob);

  noise.add_channel<cudaq::types::h>({0}, depolarization);
  noise.add_channel<cudaq::types::h>({1}, depolarization);
  noise.add_channel<cudaq::types::x>({0}, depolarization);

  cudaq::set_noise(noise);

  cudaq::ExecutionContext ctx_gen("generate_data", 2);
  auto &platform2 = cudaq::get_platform();
  ctx_gen.set_seed(41);

  platform2.set_exec_ctx(&ctx_gen);

  auto m = kernel(num_qubits, num_rounds);
  printf(" ------------------ Result:  %d\n", m);

  ctx_gen.dump_recorded_states();
  ctx_gen.dump_error_data();
  auto errors = ctx_gen.get_error_data();
  auto replay_cols = ctx_gen.get_replay_columns();
  printf("Total replay columns: %zu\n", replay_cols);
  platform2.reset_exec_ctx();
  printf("Total errors recorded: %zu\n", errors.size());
  printf("+++++++++++++++++++++++++++++++++\n");
  printf("+++++++++++++++++++++++++++++++++\n");

  cudaq::ExecutionContext ctx_rep("replay_errors", 2);
  platform2.set_exec_ctx(&ctx_rep);
  ctx_rep.set_error_data(errors);
  ctx_rep.update_replay_columns(replay_cols);
  ctx_rep.set_seed(41);

  auto m2 = kernel(num_qubits, num_rounds);
  printf(" ------------------ Result after replaying errors:  %d\n", m2);

  ctx_rep.dump_recorded_states();
  ctx_rep.dump_error_data();
  platform2.reset_exec_ctx();
  return 0;
}
