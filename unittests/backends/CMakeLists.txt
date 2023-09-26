# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Python COMPONENTS Interpreter)
if (OPENSSL_FOUND AND Python_FOUND AND CUDAQ_TEST_MOCK_SERVERS)
  add_subdirectory(oqc)
  add_subdirectory(quantinuum)
  add_subdirectory(ionq)
  add_subdirectory(iqm)
endif()
add_subdirectory(qpp_observe)
