# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

def test_pr_4039():
    
    @cudaq.kernel
    def demo_circuit_qpu(
        num_data: int,
        num_logical: int,
    ) -> int:
    
        # Allocate qubits
        data = cudaq.qvector(num_logical * num_data)
    
        # Note: this only works up to 64 bits, so a single logical qubit with distance 7.
        ret = 0
        for i in range(num_logical):
            if i > 0:
                ret = ret << num_data
            sub_data = data[i * num_data:(i + 1) * num_data]
            sub_meas = mz(sub_data)
            ret |= cudaq.to_integer(sub_meas)
    
        # The remaining bits are allocated to the number of corrections.
        return ret
    
    target_name = "quantinuum"
    emulate = True
    cudaq.set_target(target_name,
                         emulate=emulate,
                         machine="Helios-1Dummy")
    
    num_data = 9
    num_logical = 1
    num_shots = 10
    
    run_result = cudaq.run(
        demo_circuit_qpu,
        num_data,
        num_logical,
        shots_count=num_shots,
        noise_model=None)


