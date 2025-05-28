# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Add nvq++ compile + execution test of code examples
# Args:
#   TEST_NAME: name of the test executable. Test name is prefixed with "nvqpp"
#   SOURCE_LOCATION: location of the source file (relative to 'sphinx' directory by default) 
# Optional keyword args:
#   TARGET <TARGET_NAME>: name of the target to use
#   TARGET_OPTION <Option>: extra option for the target
#   SOURCE_DIR <DIR>: the directory that SOURCE_LOCATION is relative to (if not the default)
#   LAUNCH_COMMAND <COMMAND>: the command to launch the test (e.g., mpirun)
function(add_nvqpp_test TEST_NAME SOURCE_LOCATION)
  cmake_parse_arguments(PARSED_ARGS "" "TARGET;LABELS;SOURCE_DIR;LAUNCH_COMMAND;APPLICATION_ARGS;TARGET_OPTION" "" ${ARGN})
  set(NVQPP_COMPILE_ARGS "--library-mode ")
  if(PARSED_ARGS_TARGET)
    set(NVQPP_COMPILE_ARGS "${NVQPP_COMPILE_ARGS} --target ${PARSED_ARGS_TARGET}")
    if (PARSED_ARGS_TARGET_OPTION)
      set(NVQPP_COMPILE_ARGS "${NVQPP_COMPILE_ARGS} --${PARSED_ARGS_TARGET}-option ${PARSED_ARGS_TARGET_OPTION}")
    endif()
  endif()
  if (NOT PARSED_ARGS_SOURCE_DIR)
    set(PARSED_ARGS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sphinx")
  endif()
  add_test(
  NAME
    nvqpp_${TEST_NAME}
  COMMAND
    bash -c "rm -f ${TEST_NAME}; ${CMAKE_BINARY_DIR}/bin/nvq++ ${NVQPP_COMPILE_ARGS} ${PARSED_ARGS_SOURCE_DIR}/${SOURCE_LOCATION} -o ${TEST_NAME} && \
              ${PARSED_ARGS_LAUNCH_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME} ${PARSED_ARGS_APPLICATION_ARGS}"
  )
  if(PARSED_ARGS_LABELS)
    set_tests_properties(nvqpp_${TEST_NAME} PROPERTIES LABELS "${PARSED_ARGS_LABELS}")
  endif()
endfunction()

add_nvqpp_test(GHZ examples/cpp/basics/static_kernel.cpp)
add_nvqpp_test(MultiControlOps examples/cpp/basics/multi_controlled_operations.cpp)
add_nvqpp_test(ExpVals examples/cpp/basics/expectation_values.cpp)
add_nvqpp_test(MidCircuitMeasurements examples/cpp/basics/mid_circuit_measurement.cpp)
add_nvqpp_test(PhaseEstimation applications/cpp/phase_estimation.cpp)
add_nvqpp_test(Grover applications/cpp/grover.cpp)
add_nvqpp_test(QAOA applications/cpp/qaoa_maxcut.cpp)
add_nvqpp_test(VQEH2 applications/cpp/vqe_h2.cpp)
add_nvqpp_test(AmplitudeEstimation applications/cpp/amplitude_estimation.cpp)
add_nvqpp_test(Builder examples/cpp/other/builder/builder.cpp)
add_nvqpp_test(QAOABuilder examples/cpp/other/builder/qaoa_maxcut_builder.cpp)
add_nvqpp_test(VQEH2Builder examples/cpp/other/builder/vqe_h2_builder.cpp)
add_nvqpp_test(ComputeAction examples/cpp/other/compute_actions.cpp)
add_nvqpp_test(Gradients examples/cpp/other/gradients.cpp)
add_nvqpp_test(IterativePhaseEstimation applications/cpp/iterative_qpe.cpp)
add_nvqpp_test(RandomWalkPhaseEstimation applications/cpp/random_walk_qpe.cpp)

if (CUSTATEVEC_ROOT AND CUDA_FOUND) 
  add_nvqpp_test(CuQuantumGHZ examples/cpp/basics/cuquantum_backends.cpp TARGET nvidia LABELS gpu_required)  
  add_nvqpp_test(CuQuantumBernsteinVazirani applications/cpp/bernstein_vazirani.cpp TARGET nvidia LABELS gpu_required)  
endif()

# code snippets in docs
add_nvqpp_test(QuickStart_default quick_start.cpp SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)
add_nvqpp_test(FirstObserve using/first_observe.cpp SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)
add_nvqpp_test(FirstSample using/first_sample.cpp SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)
add_nvqpp_test(Timing using/time.cpp SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp APPLICATION_ARGS "10")

# Add default_ops C++ snippet tests
add_nvqpp_test(x_op sphinx/snippets/cpp/default_ops/x_op.cpp)
add_nvqpp_test(y_op sphinx/snippets/cpp/default_ops/y_op.cpp)
add_nvqpp_test(z_op sphinx/snippets/cpp/default_ops/z_op.cpp)
add_nvqpp_test(h_op sphinx/snippets/cpp/default_ops/h_op.cpp)
add_nvqpp_test(r1_op sphinx/snippets/cpp/default_ops/r1_op.cpp)
add_nvqpp_test(rx_op sphinx/snippets/cpp/default_ops/rx_op.cpp)
add_nvqpp_test(ry_op sphinx/snippets/cpp/default_ops/ry_op.cpp)
add_nvqpp_test(rz_op sphinx/snippets/cpp/default_ops/rz_op.cpp)
add_nvqpp_test(s_op sphinx/snippets/cpp/default_ops/s_op.cpp)
add_nvqpp_test(t_op sphinx/snippets/cpp/default_ops/t_op.cpp)
add_nvqpp_test(swap_op sphinx/snippets/cpp/default_ops/swap_op.cpp)
add_nvqpp_test(u3_op sphinx/snippets/cpp/default_ops/u3_op.cpp)
add_nvqpp_test(adjoint_op sphinx/snippets/cpp/default_ops/adjoint_op.cpp)
add_nvqpp_test(controlled_op sphinx/snippets/cpp/default_ops/controlled_op.cpp)
add_nvqpp_test(negated_control sphinx/snippets/cpp/default_ops/negated_control.cpp)
add_nvqpp_test(mz_op sphinx/snippets/cpp/default_ops/mz_op.cpp)
add_nvqpp_test(mx_op sphinx/snippets/cpp/default_ops/mx_op.cpp)
add_nvqpp_test(my_op sphinx/snippets/cpp/default_ops/my_op.cpp)
add_nvqpp_test(custom_op sphinx/snippets/cpp/default_ops/custom_op.cpp)
add_nvqpp_test(create_op sphinx/snippets/cpp/default_ops/create_op.cpp)
add_nvqpp_test(annihilate_op sphinx/snippets/cpp/default_ops/annihilate_op.cpp)
add_nvqpp_test(phase_shift_op sphinx/snippets/cpp/default_ops/phase_shift_op.cpp)
add_nvqpp_test(beam_splitter_op sphinx/snippets/cpp/default_ops/beam_splitter_op.cpp)
add_nvqpp_test(mz_qumode_op sphinx/snippets/cpp/default_ops/mz_qumode_op.cpp)
add_nvqpp_test(custom_kraus_channel sphinx/snippets/cpp/noise/custom_kraus_channel.cpp)
add_nvqpp_test(apply_noise_example sphinx/snippets/cpp/noise/apply_noise_example.cpp)
add_nvqpp_test(sample_bell_state_cpp_algo sphinx/snippets/cpp/algorithmic/sample_bell_state.cpp)
add_nvqpp_test(sample_explicit_measurements_cpp_algo sphinx/snippets/cpp/algorithmic/sample_explicit_measurements.cpp)
add_nvqpp_test(sample_implicit_measurements_dump_cpp_algo sphinx/snippets/cpp/algorithmic/sample_implicit_measurements_dump.cpp)
add_nvqpp_test(sample_with_measurements_global_reg_cpp_algo sphinx/snippets/cpp/algorithmic/sample_with_measurements_global_reg.cpp)
add_nvqpp_test(observe_result_usage_cpp_algo sphinx/snippets/cpp/algorithmic/observe_result_usage.cpp)
add_nvqpp_test(observe_h2_ansatz_example_cpp_algo sphinx/snippets/cpp/algorithmic/observe_h2_ansatz_example.cpp)
add_nvqpp_test(optimizer_cobyla_example_cpp_algo sphinx/snippets/cpp/algorithmic/optimizer_cobyla_example.cpp)
add_nvqpp_test(optimizer_lbfgs_gradient_example_cpp_algo sphinx/snippets/cpp/algorithmic/optimizer_lbfgs_gradient_example.cpp)
add_nvqpp_test(example_hello_bell_state_cpp sphinx/snippets/cpp/examples/hello_bell_state.cpp)
add_nvqpp_test(example_ghz_preparation_sampling_cpp sphinx/snippets/cpp/examples/ghz_preparation_sampling.cpp)
add_nvqpp_test(example_deuteron_energy_sweep_cpp sphinx/snippets/cpp/examples/deuteron_energy_sweep.cpp)
add_nvqpp_test(kernels_types_example_cpp sphinx/snippets/cpp/kernels/kernels_types_example.cpp)
add_nvqpp_test(kernels_allowed_types_cpp sphinx/snippets/cpp/kernels/kernels_allowed_types.cpp)
add_nvqpp_test(kernels_variable_allocation_cpp sphinx/snippets/cpp/kernels/kernels_variable_allocation.cpp)
add_nvqpp_test(kernels_pass_by_value_cpp sphinx/snippets/cpp/kernels/kernels_pass_by_value.cpp)
add_nvqpp_test(kernels_variable_capture_cpp sphinx/snippets/cpp/kernels/kernels_variable_capture.cpp)
add_nvqpp_test(kernels_composition_templates_cpp sphinx/snippets/cpp/kernels/kernels_composition_templates.cpp)
add_nvqpp_test(kernels_composition_constraints_cpp sphinx/snippets/cpp/kernels/kernels_composition_constraints.cpp)


set(NGPUS 0)
if (CUSTATEVEC_ROOT AND CUDA_FOUND) 
  add_nvqpp_test(QuickStart_nvidia quick_start.cpp TARGET nvidia LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)

  # mqpu snippets need custatevec backend and optionally MPI
  add_nvqpp_test(SampleAsync using/cudaq/platform/sample_async.cpp TARGET nvidia TARGET_OPTION mqpu LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)
  add_nvqpp_test(ObserveMQPU using/cudaq/platform/observe_mqpu.cpp TARGET nvidia TARGET_OPTION mqpu LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)
  add_nvqpp_test(StateAsyncMQPU using/cudaq/platform/get_state_async.cpp TARGET nvidia TARGET_OPTION mqpu LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)

  # Legacy check for the `nvidia-mqpu` target
  add_nvqpp_test(LegacySampleAsync using/cudaq/platform/sample_async.cpp TARGET nvidia-mqpu LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)
  add_nvqpp_test(LegacyObserveMQPU using/cudaq/platform/observe_mqpu.cpp TARGET nvidia-mqpu LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)
  add_nvqpp_test(LegacyStateAsyncMQPU using/cudaq/platform/get_state_async.cpp TARGET nvidia-mqpu LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)

  # Add the MPI test if MPI was found and there are more than 2 GPUs
  if (MPI_CXX_FOUND)
    # Count the number of GPUs
    find_program(NVIDIA_SMI "nvidia-smi")
    if(NVIDIA_SMI)
      execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
      # Only build this test if we have more than 1 GPU
      if (${NGPUS} GREATER_EQUAL 2)
        add_nvqpp_test(ObserveMQPU_MPI using/cudaq/platform/observe_mqpu_mpi.cpp
                        TARGET nvidia 
                        TARGET_OPTION mqpu 
                        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp 
                        LAUNCH_COMMAND "${MPIEXEC} --allow-run-as-root -np 2")
        add_nvqpp_test(LegacyObserveMQPU_MPI using/cudaq/platform/observe_mqpu_mpi.cpp
                        TARGET nvidia-mqpu 
                        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp 
                        LAUNCH_COMMAND "${MPIEXEC} --allow-run-as-root -np 2")
      endif()
    endif(NVIDIA_SMI)  
  endif()   
endif()

add_nvqpp_test(photonics_sim targets/cpp/photonics.cpp TARGET orca-photonics)
add_nvqpp_test(SampleAsyncRemote using/cudaq/platform/sample_async_remote.cpp TARGET remote-mqpu SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp)
set_tests_properties(
  nvqpp_SampleAsyncRemote
    PROPERTIES
      ENVIRONMENT_MODIFICATION PATH=path_list_append:${CMAKE_BINARY_DIR}/bin
)

# Only add the python tests if we built the python API
if (CUDAQ_ENABLE_PYTHON)
  function(add_pycudaq_test TEST_NAME SOURCE_LOCATION)
    cmake_parse_arguments(PARSED_ARGS "" "LABELS;SOURCE_DIR;LAUNCH_COMMAND" "" ${ARGN}) 
    if (NOT PARSED_ARGS_SOURCE_DIR)
      set(PARSED_ARGS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sphinx/examples/python")
    endif()
    add_test(
    NAME
      pycudaq_${TEST_NAME}
    COMMAND
      bash -c "${PARSED_ARGS_LAUNCH_COMMAND} ${Python_EXECUTABLE} ${PARSED_ARGS_SOURCE_DIR}/${SOURCE_LOCATION}"
    )
    set_tests_properties(pycudaq_${TEST_NAME} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
    if(PARSED_ARGS_LABELS)
      set_tests_properties(pycudaq_${TEST_NAME} PROPERTIES LABELS "${PARSED_ARGS_LABELS}")
    endif()
  endfunction()

  add_pycudaq_test(Intro intro.py)
  if (CUDA_FOUND)
    add_pycudaq_test(EvolveDynamics using/backends/dynamics.py LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  endif()

  if (CUTENSORNET_ROOT AND CUDA_FOUND) 
    # This example uses tensornet backend.
    add_pycudaq_test(SampleAsyncRemote using/cudaq/platform/sample_async_remote.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  endif()
  
  if (CUSTATEVEC_ROOT AND CUDA_FOUND) 
    add_pycudaq_test(SampleAsync using/cudaq/platform/sample_async.py LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
    add_pycudaq_test(ObserveMQPU using/cudaq/platform/observe_mqpu.py LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
    if (MPI_CXX_FOUND AND ${NGPUS} GREATER_EQUAL 2)
      add_pycudaq_test(ObserveMQPU_MPI using/cudaq/platform/observe_mqpu_mpi.py 
                        LABELS "gpu_required;mgpus_required"
                        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python
                        LAUNCH_COMMAND "${MPIEXEC} --allow-run-as-root -np 2")
    endif()
  endif()

  # Add default_ops Python snippet tests
  add_pycudaq_test(x_op using/cudaq/default_ops/x_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(y_op using/cudaq/default_ops/y_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(z_op using/cudaq/default_ops/z_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(h_op using/cudaq/default_ops/h_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(r1_op using/cudaq/default_ops/r1_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(rx_op using/cudaq/default_ops/rx_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(ry_op using/cudaq/default_ops/ry_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(rz_op using/cudaq/default_ops/rz_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(s_op using/cudaq/default_ops/s_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(t_op using/cudaq/default_ops/t_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(swap_op using/cudaq/default_ops/swap_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(u3_op using/cudaq/default_ops/u3_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(adjoint_op using/cudaq/default_ops/adjoint_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(controlled_op using/cudaq/default_ops/controlled_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(mz_op using/cudaq/default_ops/mz_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(mx_op using/cudaq/default_ops/mx_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(my_op using/cudaq/default_ops/my_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(custom_op using/cudaq/default_ops/custom_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(create_op using/cudaq/default_ops/create_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(annihilate_op using/cudaq/default_ops/annihilate_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(phase_shift_op using/cudaq/default_ops/phase_shift_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(beam_splitter_op using/cudaq/default_ops/beam_splitter_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(mz_qumode_op using/cudaq/default_ops/mz_qumode_op.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(custom_noise_channel noise/custom_noise_channel.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(apply_noise_example noise/apply_noise_example.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(sample_bell_state_py_algo algorithmic/sample_bell_state.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(sample_explicit_measurements_py_algo algorithmic/sample_explicit_measurements.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(sample_implicit_measurements_dump_py_algo algorithmic/sample_implicit_measurements_dump.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(sample_with_measurements_global_reg_py_algo algorithmic/sample_with_measurements_global_reg.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(observe_result_usage_py_algo algorithmic/observe_result_usage.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(observe_h2_ansatz_example_py_algo algorithmic/observe_h2_ansatz_example.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(optimizer_cobyla_example_py_algo algorithmic/optimizer_cobyla_example.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(optimizer_lbfgs_gradient_example_py_algo algorithmic/optimizer_lbfgs_gradient_example.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(example_hello_bell_state_py examples/hello_bell_state.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(example_ghz_preparation_sampling_py examples/ghz_preparation_sampling.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(example_qpe_t_gate_py examples/qpe_t_gate.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(example_deuteron_energy_sweep_py examples/deuteron_energy_sweep.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(example_grover_3_qubit_oracle_py examples/grover_3_qubit_oracle.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(kernels_types_example_py kernels/kernels_types_example.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(kernels_allowed_types_py kernels/kernels_allowed_types.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(kernels_variable_allocation_py kernels/kernels_variable_allocation.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(kernels_pass_by_value_py kernels/kernels_pass_by_value.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(kernels_variable_capture_py kernels/kernels_variable_capture.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
  add_pycudaq_test(kernels_composition_callable_py kernels/kernels_composition_callable.py SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
endif()