# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(METIS)
if (METIS_FOUND)
  message(STATUS "Metis found, building MetisGraphPartitioner for QuakeCircuitCutPass.")
  add_subdirectory(metis)
endif()

