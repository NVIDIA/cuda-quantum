# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The CLI reports machine-readable capabilities, including the executable oracles
# with each oracle's assurance tier. Two exact tiers are supported: the dense
# exact-unitary one and the Clifford-only exact-clifford-sim one, which trades
# domain for reach past the dense qubit bound.

# RUN: PYTHONPATH=%cudaq_python_root python3 -m cudaq._compiler --capabilities | FileCheck %s

# CHECK-DAG: "assurance_tiers": [
# CHECK-DAG: "exact-unitary"
# CHECK-DAG: "exact-clifford-sim"

# CHECK-DAG: "kind": "strict-unitary"
# CHECK-DAG: "kind": "up-to-global-phase"
# CHECK-DAG: "kind": "clifford-tableau"
