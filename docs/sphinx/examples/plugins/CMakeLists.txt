###############################################################################
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
###############################################################################

add_custom_target(cudaq-example-plugins)

add_subdirectory(mock_rest)
add_subdirectory(trace_qpu)

add_dependencies(cudaq-example-plugins
  cudaq-example-serverhelper-mock_rest
  cudaq-example-qpu-trace_qpu)
