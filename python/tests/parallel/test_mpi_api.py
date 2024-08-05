# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, pytest, importlib
import cudaq

skipIfUnsupported = pytest.mark.skipif(importlib.util.find_spec('mpi4py')
                                       is None,
                                       reason="mpi4py not found")


@skipIfUnsupported
def testMPI():
    cudaq.mpi.initialize()
    assert cudaq.mpi.is_initialized() == True
    # Check rank API
    if os.environ.get('OMPI_COMM_WORLD_RANK') is not None:
        print("Rank:", os.environ.get('OMPI_COMM_WORLD_RANK'))
        assert cudaq.mpi.rank() == int(os.environ.get('OMPI_COMM_WORLD_RANK'))

    if os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        assert cudaq.mpi.num_ranks() == int(
            os.environ.get('OMPI_COMM_WORLD_SIZE'))

    # all_gather integers
    localData = [cudaq.mpi.rank()]
    gatherData = cudaq.mpi.all_gather(cudaq.mpi.num_ranks(), localData)
    assert len(gatherData) == cudaq.mpi.num_ranks()
    for idx, x in enumerate(gatherData):
        assert x == idx

    # all_gather floats
    localData = [float(cudaq.mpi.rank())]
    gatherData = cudaq.mpi.all_gather(cudaq.mpi.num_ranks(), localData)
    assert len(gatherData) == cudaq.mpi.num_ranks()
    for idx, x in enumerate(gatherData):
        assert abs(gatherData[idx] - float(idx)) < 1e-12

    # Broadcast
    ref_data = [1.0, 2.0, 3.0]
    if cudaq.mpi.rank() == 0:
        data = ref_data
    else:
        data = []

    data = cudaq.mpi.broadcast(data, len(ref_data), 0)
    for idx, x in enumerate(data):
        assert abs(x - ref_data[idx]) < 1e-12

    cudaq.mpi.finalize()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
