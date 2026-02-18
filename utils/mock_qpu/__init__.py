# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .preallocated_qubits_context import PreallocatedQubitsContext

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


def start_server(backend: str):
    """Start the mock QPU backend server."""

    import cudaq
    import uvicorn

    port = get_backend_port(backend)

    match backend:
        case "anyon":
            from .anyon import app
        case "braket":
            from .braket import app
        case "infleqtion":
            from .infleqtion import app
        case "ionq":
            from .ionq import app
        case "iqm":
            from .iqm import app
        case "oqc":
            from .oqc import app
        case "qci":
            from .qci import app
        case "quantinuum":
            from .quantinuum import app
        case "quantum_machines":
            from .quantum_machines import app
        case _:
            # <backend> is in all_backend_names() but not handled!
            raise ValueError(
                f"case '{backend}' is not handled in start_mock_qpu.py")

    cudaq.set_random_seed(13)

    print(f"Starting {backend} server on port {port}")
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")
