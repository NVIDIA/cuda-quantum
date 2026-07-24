# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The CLI reports machine-readable capabilities, including the executable oracles
# with each oracle's assurance tier. Only the exact-unitary tier is supported.

# RUN: PYTHONPATH=%cudaq_python_root python3 -m cudaq._compiler --capabilities | FileCheck %s

# CHECK-DAG: "assurance_tiers": [
# CHECK-DAG: "exact-unitary"

# CHECK-DAG: "kind": "strict-unitary"
# CHECK-DAG: "kind": "up-to-global-phase"
