# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Simulates a Python package that re-exports C++ device kernels from a C++
# extension module (issue #2348). The kernels are registered under
# cudaq_test_cpp_algo, but are accessed here via the qlib namespace.
from cudaq_test_cpp_algo import qstd
