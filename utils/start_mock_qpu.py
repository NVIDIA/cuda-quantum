#!/usr/bin/env python3

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import argparse

from mock_qpu import get_backend_port, all_backend_names


def start_server(backend: str):
    """Start the mock QPU backend server."""

    import cudaq
    import uvicorn

    port = get_backend_port(backend)

    match backend:
        case "anyon":
            from mock_qpu.anyon import app
        case "braket":
            from mock_qpu.braket import app
        case "infleqtion":
            from mock_qpu.infleqtion import app
        case "ionq":
            from mock_qpu.ionq import app
        case "iqm":
            from mock_qpu.iqm import app
        case "oqc":
            from mock_qpu.oqc import app
        case "qci":
            from mock_qpu.qci import app
        case "quantinuum":
            from mock_qpu.quantinuum import app
        case "quantum_machines":
            from mock_qpu.quantum_machines import app
        case _:
            # <backend> is in all_backend_names() but not handled!
            raise ValueError(
                f"case '{backend}' is not handled in start_mock_qpu.py")

    cudaq.set_random_seed(13)

    print(f"Starting {backend} server on port {port}")
    uvicorn.run(app, port=port, host='0.0.0.0', log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start one of the available mock QPU backend server.")
    parser.add_argument("backend",
                        type=str,
                        help="backend name",
                        choices=all_backend_names())
    args = parser.parse_args()
    backend = args.backend.lower()

    start_server(backend)
