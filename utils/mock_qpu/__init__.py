# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Default ports for each mock QPU backend
MOCK_QPU_PORTS = {
    "quantinuum": 62440,
    "ionq": 62441,
    "oqc": 62442,
    "iqm": 62443,
    "braket": 62445,
    "anyon": 62446,
    "infleqtion": 62447,
    "quantum_machines": 62448,
    "qci": 62449
}


def get_backend_port(backend: str) -> int:
    return MOCK_QPU_PORTS[backend]


def all_backend_names() -> list[str]:
    return list(MOCK_QPU_PORTS.keys())
