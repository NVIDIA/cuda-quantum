# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Python COMPONENTS Interpreter)
if (OPENSSL_FOUND AND CUDAQ_ENABLE_PYTHON AND CUDAQ_TEST_MOCK_SERVERS)
  add_subdirectory(anyon)
  add_subdirectory(braket)
  add_subdirectory(infleqtion)
  add_subdirectory(ionq)
  add_subdirectory(iqm)
  add_subdirectory(oqc)
  add_subdirectory(quantinuum)
  if (CUDAQ_ENABLE_QUANTUM_MACHINES_BACKEND)
    add_subdirectory(quantum_machines)
  endif()
endif()
add_subdirectory(pasqal)
add_subdirectory(qpp_observe)
add_subdirectory(quera)
add_subdirectory(qilimanjaro)
