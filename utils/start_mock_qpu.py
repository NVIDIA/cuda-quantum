# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys

import cudaq
from mock_qpu import get_backend_port, all_backend_names
import uvicorn


def main():
    if len(sys.argv) < 2:
        print("Please specify a backend: ", ", ".join(all_backend_names()))
        sys.exit(1)

    backend = sys.argv[1].lower()

    try:
        port = get_backend_port(backend)
    except KeyError:
        print(f"Unknown backend '{backend}'. Valid options: ",
              ", ".join(all_backend_names()))
        sys.exit(1)

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
    main()
