# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest
import os
from typing import List


@pytest.fixture(scope="function", autouse=True)
def configureTarget():
    # Set the targeted QPU
    cudaq.set_target('ionq', emulate='true')

    yield "Running the tests."

    cudaq.reset_target()


def test_Ionq_cudaq_uccsd():
    repro_num_electrons = 2
    repro_num_qubits = 8

    # # should be 3 thetas
    repro_thetas = [
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558
    ]

    @cudaq.kernel
    def repro_trial_state(qubits: cudaq.qvector, num_electrons: int,
                          num_qubits: int, thetas: List[float]):
        for i in range(num_electrons):
            x(qubits[i])
        uccsd(qubits, thetas, num_electrons, num_qubits)

    @cudaq.kernel
    def repro():
        repro_qubits = cudaq.qvector(repro_num_qubits)
        repro_trial_state(repro_qubits, repro_num_electrons, repro_num_qubits,
                          repro_thetas)

    counts = cudaq.sample(repro, shots_count=1000)
    assert len(counts) == 6
    assert '00000011' in counts
    assert '00000110' in counts
    assert '00010010' in counts
    assert '01000010' in counts
    assert '10000001' in counts
    assert '11000000' in counts


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
