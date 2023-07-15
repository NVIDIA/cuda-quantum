# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

function(add_nvqpp_test TEST_NAME SOURCE_LOCATION)
  add_test(
  NAME
    nvqpp_${TEST_NAME}
  COMMAND
    bash -c "${CMAKE_BINARY_DIR}/bin/nvq++ ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/examples/cpp/${SOURCE_LOCATION} -o ${TEST_NAME} ;\
              ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}"
  )
endfunction()

add_nvqpp_test(GHZ basics/static_kernel.cpp)
add_nvqpp_test(MultiControlOps basics/multi_controlled_operations.cpp)
add_nvqpp_test(ExpVals basics/expectation_values.cpp)
# We need quake to help with mid-circuit measurements
add_nvqpp_test(MidCircuitMeasurements basics/mid_circuit_measurement.cpp)
add_nvqpp_test(PhaseEstimation algorithms/phase_estimation.cpp)
add_nvqpp_test(Grover algorithms/grover.cpp)
add_nvqpp_test(QAOA algorithms/qaoa_maxcut.cpp)
add_nvqpp_test(VQEH2 algorithms/vqe_h2.cpp)
add_nvqpp_test(AmplitudeEstimation algorithms/amplitude_estimation.cpp)
add_nvqpp_test(Builder other/builder/builder.cpp)
add_nvqpp_test(QAOABuilder other/builder/qaoa_maxcut_builder.cpp)
add_nvqpp_test(VQEH2Builder other/builder/vqe_h2_builder.cpp)
add_nvqpp_test(ComputeAction other/compute_actions.cpp)
add_nvqpp_test(Gradients other/gradients.cpp)
add_nvqpp_test(IterativePhaseEstimation other/iterative_qpe.cpp)
add_nvqpp_test(RandomWalkPhaseEstimation other/random_walk_qpe.cpp)

if (CUSTATEVEC_ROOT AND CUDA_FOUND) 
  add_test(
    NAME
      nvqpp_cuquantum
    COMMAND
      bash -c "${CMAKE_BINARY_DIR}/bin/nvq++ --target nvidia ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/examples/cpp/basics/cuquantum_backends.cpp -o CuQuantumBackend ;\
                ${CMAKE_CURRENT_BINARY_DIR}/CuQuantumBackend"
    )
endif()

# Only add the python tests if we built the python API
if (NOT CUDAQ_ENABLE_PYTHON)
  return()
endif()

# And we have the python interpreter
find_package(Python COMPONENTS Interpreter QUIET)
if (NOT Python_FOUND)
  return()
endif()

function(add_pycudaq_test TEST_NAME SOURCE_LOCATION)
  add_test(
  NAME
    pycudaq_${TEST_NAME}
  COMMAND
    ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sphinx/examples/python/${SOURCE_LOCATION}
  )
endfunction()

add_pycudaq_test(Intro intro.py)
add_pycudaq_test(BernsteinVazirani bernstein_vazirani.py)
add_pycudaq_test(QAOA qaoa_maxcut.py)
add_pycudaq_test(VQE simple_vqe.py)
add_pycudaq_test(VQEAdvanced advanced_vqe.py)

