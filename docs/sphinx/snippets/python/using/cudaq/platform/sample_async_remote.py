# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
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
    kernel, nr_controls = cudaq.make_kernel(int)
    controls = kernel.qalloc(nr_controls)
    targets = kernel.qalloc(2)
    kernel.h(controls)
    for tidx in range(2):
        kernel.cx(controls, targets[tidx])
    kernel.mz(controls)
    kernel.mz(targets)

    # Set the target to execute on and query the number of QPUs in the system;
    # The number of QPUs is equal to the number of (auto-)launched server instances.
    cudaq.set_target("remote-mqpu",
                     backend=backend,
                     auto_launch=str(servers) if servers.isdigit() else "",
                     url="" if servers.isdigit() else servers)
    num_qpus = cudaq.get_target().num_qpus()
    print("Number of virtual QPUs:", num_qpus)

    # We will launch asynchronous sampling tasks,
    # and will store the results as a future we can query at some later point.
    # Each QPU (indexed by an unique Id) is associated with a remote REST server.
    count_futures = []
    for i in range(num_qpus):
        count_futures.append(cudaq.sample_async(kernel, i + 1, qpu_id=i))
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
