# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

The following tests are disabled because sandbox only supports 12 qubits and the
test tries to use more.
- test/NVQPP/graph_coloring-1.cpp
- test/NVQPP/graph_coloring.cpp

The following test is disabled due to https://github.com/NVIDIA/cuda-quantum/issues/695
- test/NVQPP/cudaq_observe.cpp
