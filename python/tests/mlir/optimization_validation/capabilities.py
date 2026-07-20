# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The CLI reports machine-readable capabilities, including the oracle roadmap
# with each oracle's assurance tier. Only the exact-tier oracles are supported.

# RUN: PYTHONPATH=%cudaq_python_root python3 -m cudaq._compiler.optimization_cli --capabilities | FileCheck %s

# CHECK-DAG: "capability_schema_version": 2
# CHECK-DAG: "assurance_tiers": [
# CHECK-DAG: "exact"

# CHECK-DAG: "kind": "up-to-global-phase"
# CHECK-DAG: "kind": "clifford-tableau"
# CHECK-DAG: "tier": "scalable-exact"
# CHECK-DAG: "kind": "density-matrix"
# CHECK-DAG: "tier": "mixed-state"
# CHECK-DAG: "kind": "statevector-expectation"
# CHECK-DAG: "tier": "advisory"
