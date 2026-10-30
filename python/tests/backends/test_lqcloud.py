# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from collections import Counter
from multiprocessing import Process
import os

import cudaq
import pytest
from network_utils import check_server_connection
from utils.mock_qpu.lqcloud import startServer, TEST_API_KEY

skipIfLQCloudNotInstalled = pytest.mark.skipif(
    not cudaq.has_target("lqcloud"),
    reason="Could not find lqcloud in installation")
pytestmark = [
    skipIfLQCloudNotInstalled,
    pytest.mark.xdist_group("lqcloud_mock")
]

PORT = 62453


@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    previous_key = os.environ.get("LQCLOUD_API_KEY")
    os.environ["LQCLOUD_API_KEY"] = TEST_API_KEY
    server = Process(target=startServer, args=(PORT,))
    server.start()
    try:
        assert check_server_connection(PORT), "LQCloud mock did not start"
        cudaq.set_target("lqcloud",
                         url=f"http://localhost:{PORT}",
                         qpu="mock_qpu")
        yield
    finally:
        server.terminate()
        server.join()
        cudaq.reset_target()
        if previous_key is None:
            os.environ.pop("LQCLOUD_API_KEY", None)
        else:
            os.environ["LQCLOUD_API_KEY"] = previous_key


def test_lqcloud_sample():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])

    for result in (cudaq.sample(kernel, shots_count=400),
                   cudaq.sample_async(kernel, shots_count=400).get()):
        assert result.get_total_shots() == 400
        assert set(dict(result.items())) == {"00", "11"}
        assert all(140 <= count <= 260 for _, count in result.items())

    @cudaq.kernel
    def routed() -> list[bool]:
        q = cudaq.qvector(3)
        barrier(q)
        x(q[0])
        first = mz(q[0])
        barrier(q)
        z.ctrl(q[0], q[2])
        barrier(q)
        x(q[0])
        second = mz(q[0])
        x(q[2])
        third = mz(q[2])
        fourth = mz(q[1])
        return [first, second, third, fourth]

    for records in (cudaq.run(routed, shots_count=20),
                    cudaq.run_async(routed, shots_count=20).get()):
        assert len(records) == 20
        assert all(record == [True, False, True, False] for record in records)

    @cudaq.kernel
    def collapse() -> list[bool]:
        q = cudaq.qvector(1)
        h(q[0])
        first = mz(q[0])
        h(q[0])
        second = mz(q[0])
        return [first, second]

    records = cudaq.run(collapse, shots_count=400)
    counts = Counter(tuple(record) for record in records)
    assert len(records) == 400
    assert set(counts) == {(False, False), (False, True), (True, False),
                           (True, True)}
    assert all(50 <= count <= 150 for count in counts.values())

    @cudaq.kernel
    def long_history() -> list[bool]:
        q = cudaq.qvector(1)
        x(q[0])
        return [mz(q[0]) for _ in range(489)]

    records = cudaq.run_async(long_history, shots_count=20).get()
    assert len(records) == 20
    assert all(record == [True] * 489 for record in records)

    builder = cudaq.make_kernel()
    q = builder.qalloc(1)
    with pytest.raises(ValueError, match="non-empty static qvector"):
        builder.barrier(q[0])

    # A later default request must still use the normal terminal result order.
    result = cudaq.sample(kernel, shots_count=400)
    assert result.get_total_shots() == 400
    assert set(dict(result.items())) == {"00", "11"}
    assert all(140 <= count <= 260 for _, count in result.items())
