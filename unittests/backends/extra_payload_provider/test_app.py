# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
import sys
import ctypes

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise (Exception('Too few command-line arguments'))
    provider_lib = sys.argv[1]
    print(f"Loading provider library from {provider_lib}")
    lib = ctypes.cdll.LoadLibrary(provider_lib)

# Set the target
cudaq.set_target('horizon',
                 url='http://localhost:62450',
                 extra_payload_provider='sunrise')


@cudaq.kernel
def simple():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    mz(qubits)


try:
    counts = cudaq.sample(simple)
except Exception as e:
    print(e)
    sys.exit(1)
