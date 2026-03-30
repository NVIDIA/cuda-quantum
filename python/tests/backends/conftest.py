# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest
import os
from multiprocessing import Process

# ---------------------------------------------------------------------------
# Shared Quantinuum mock-server fixture
# ---------------------------------------------------------------------------
# Three test files (test_Quantinuum_{kernel,ng_kernel,builder}.py) need the
# same mock Quantinuum server and fake credentials file.  Under pytest-xdist
# each worker is a separate process, so a session-scoped fixture is
# per-worker.  We mark those files with
#   pytestmark = pytest.mark.xdist_group("quantinuum_mock")
# so xdist keeps them on a single worker sharing one server instance.
# ---------------------------------------------------------------------------

QUANTINUUM_MOCK_PORT = 62440
QUANTINUUM_CREDS_FILE = os.path.join(os.environ["HOME"],
                                     "QuantinuumFakeConfig.config")


@pytest.fixture(scope="session")
def quantinuum_mock_server():
    """Start the Quantinuum mock server and write a fake credentials file.

    Yields the path to the credentials file.
    """
    from network_utils import check_server_connection
    try:
        from utils.mock_qpu.quantinuum import startServer
    except Exception:
        pytest.skip("Mock qpu not available.", allow_module_level=False)

    with open(QUANTINUUM_CREDS_FILE, 'w') as f:
        f.write('key: {}\nrefresh: {}\ntime: 0'.format("nexus_key",
                                                       "nexus_refresh"))

    cudaq.set_random_seed(13)

    p = Process(target=startServer, args=(QUANTINUUM_MOCK_PORT,))
    p.start()

    if not check_server_connection(QUANTINUUM_MOCK_PORT):
        p.terminate()
        pytest.exit("Mock server did not start in time, skipping tests.",
                    returncode=1)

    yield QUANTINUUM_CREDS_FILE

    p.terminate()
    try:
        os.remove(QUANTINUUM_CREDS_FILE)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Shared Quantinuum LocalEmulation credentials fixture
# ---------------------------------------------------------------------------
# Two test files (test_Quantinuum_LocalEmulation_{kernel,builder}.py) share
# $HOME/FakeConfig2.config.  We use xdist_group to keep them on one worker.
# ---------------------------------------------------------------------------

QUANTINUUM_EMULATION_CREDS_FILE = os.path.join(os.environ["HOME"],
                                               "FakeConfig2.config")


@pytest.fixture(scope="function")
def quantinuum_emulation_creds():
    """Write a fake credentials file for Quantinuum local emulation tests.

    Yields the path; cleans up afterwards.
    """
    with open(QUANTINUUM_EMULATION_CREDS_FILE, 'w') as f:
        f.write('key: {}\nrefresh: {}\ntime: 0'.format("hello", "rtoken"))

    yield QUANTINUUM_EMULATION_CREDS_FILE

    try:
        os.remove(QUANTINUUM_EMULATION_CREDS_FILE)
    except FileNotFoundError:
        pass
