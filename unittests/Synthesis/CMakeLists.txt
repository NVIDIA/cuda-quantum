# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(NOT TARGET cudaq-synth)
  message(STATUS "Synthesis unit tests skipped: cudaq-synth target not built")
  return()
endif()

include(GoogleTest)

function(add_synthesis_test target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PRIVATE gtest_main cudaq-synth)
  target_include_directories(${target_name}
    PRIVATE ${CMAKE_SOURCE_DIR}/lib/Synthesis
  )
  gtest_discover_tests(${target_name}
    PROPERTIES LABELS "cudaq-synth-unittest"
    DISCOVERY_TIMEOUT 120
  )
endfunction()

add_synthesis_test(synthesis_pauli_product_test Circuit/PauliProductTest.cpp)
add_synthesis_test(synthesis_tableau_test       Circuit/TableauTest.cpp)
add_synthesis_test(synthesis_integer_test       Math/IntegerTest.cpp)
add_synthesis_test(synthesis_real_test          Math/RealTest.cpp)
add_synthesis_test(synthesis_odgp_test          Math/Grid/OdgpTest.cpp)
add_synthesis_test(synthesis_zsqrt2_test        Math/Ring/Zsqrt2Test.cpp)
add_synthesis_test(synthesis_bitvector_test     Support/BitVectorTest.cpp)
add_synthesis_test(synthesis_generator_test     Support/GeneratorTest.cpp)
add_synthesis_test(synthesis_ag_synth_test      Synthesis/AgSynthTest.cpp)
add_synthesis_test(synthesis_gridsynth_test     Synthesis/GridsynthTest.cpp)

add_custom_target(check-cudaq-synth-unittests
  COMMAND ${CMAKE_CTEST_COMMAND}
          --test-dir ${CMAKE_BINARY_DIR}
          -L cudaq-synth-unittest
          --output-on-failure
  DEPENDS
    synthesis_pauli_product_test
    synthesis_tableau_test
    synthesis_integer_test
    synthesis_real_test
    synthesis_odgp_test
    synthesis_zsqrt2_test
    synthesis_bitvector_test
    synthesis_generator_test
    synthesis_ag_synth_test
    synthesis_gridsynth_test
  COMMENT "Running all cudaq-synth unit tests"
)
