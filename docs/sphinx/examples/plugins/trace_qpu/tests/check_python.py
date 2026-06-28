# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: env PYTHONPATH=%cudaq_target_dir/../python python3 %s %cudaq_example_plugins_dir/trace-qpu

import argparse

import cudaq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend_path", nargs="?")
    args = parser.parse_args()

    if args.backend_path:
        cudaq.register_backend_path(args.backend_path)

    assert cudaq.has_target("trace_qpu")
    cudaq.set_target("trace_qpu")
    try:
        kernel = cudaq.make_kernel()
        qubit = kernel.qalloc(1)
        kernel.h(qubit[0])
        kernel.mz(qubit)

        result = cudaq.sample(kernel, shots_count=4)
        assert result.get_total_shots() == 4
    finally:
        cudaq.reset_target()


if __name__ == "__main__":
    main()
