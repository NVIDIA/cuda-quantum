# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import argparse
import cudaq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',
                        type=str,
                        required=False,
                        default="tensornet")
    parser.add_argument('--servers', type=str, required=False, default='2')
    args = parser.parse_args()

    # [Begin Documentation]
    # Specified as program input, e.g.
    # ```
    # backend = "tensornet"; servers = "2"
    # ```
    backend = args.backend
    servers = args.servers

    # Define a kernel to be sampled.
    @cudaq.kernel
    def kernel(controls_count: int):
        controls = cudaq.qvector(controls_count)
        targets = cudaq.qvector(2)
        # Place controls in superposition state.
        h(controls)
        for target in range(2):
            x.ctrl(controls, targets[target])
        # Measure.
        mz(controls)
        mz(targets)

    # Set the target to execute on and query the number of QPUs in the system;
    # The number of QPUs is equal to the number of (auto-)launched server instances.
    cudaq.set_target("remote-mqpu",
                     backend=backend,
                     auto_launch=str(servers) if servers.isdigit() else "",
                     url="" if servers.isdigit() else servers)
    qpu_count = cudaq.get_target().num_qpus()
    print("Number of virtual QPUs:", qpu_count)

    # We will launch asynchronous sampling tasks,
    # and will store the results as a future we can query at some later point.
    # Each QPU (indexed by an unique Id) is associated with a remote REST server.
    count_futures = []
    for i in range(qpu_count):

        result = cudaq.sample_async(kernel, i + 1, qpu_id=i)
        count_futures.append(result)
    print("Sampling jobs launched for asynchronous processing.")

    # Go do other work, asynchronous execution of sample tasks on-going.
    # Get the results, note future::get() will kick off a wait
    # if the results are not yet available.
    for idx in range(len(count_futures)):
        counts = count_futures[idx].get()
        print(counts)
        # [End Documentation]

        assert len(counts) == 1 << (idx + 1)
        for key in counts:
            assert len(key) == 3 + idx
