# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
if (CUDENSITYMAT_ROOT AND CUDA_FOUND) 
  add_nvqpp_test(Dynamics_Snippets using/backends/dynamics.cpp SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp/ TARGET dynamics LABELS gpu_required)  
  add_nvqpp_test(Dynamics_State_Batching_Snippets using/backends/dynamics_state_batching.cpp SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp/ TARGET dynamics LABELS gpu_required)  
  add_nvqpp_test(Dynamics_Operator_Batching_Snippets using/backends/dynamics_operator_batching.cpp SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/cpp/ TARGET dynamics LABELS gpu_required)  
endif()

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
    add_pycudaq_test(EvolveDynamicsStateBatching using/backends/dynamics_state_batching.py LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
    add_pycudaq_test(EvolveDynamicsOperatorBatching using/backends/dynamics_operator_batching.py LABELS gpu_required SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/snippets/python)
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
endif()
