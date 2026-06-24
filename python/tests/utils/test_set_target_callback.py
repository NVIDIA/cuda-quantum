# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import os
import pytest


def test_register_callback():
    cudaq.set_target("density-matrix-cpu")

    sim_name = ""  # to be updated by the callback

    def callback(target):
        nonlocal sim_name
        sim_name = target.name

    cudaq.register_set_target_callback(callback, "my_callback")
    assert sim_name == "density-matrix-cpu"

    # Register another one
    called = False

    def another_callback(target):
        nonlocal called
        called = True

    cudaq.register_set_target_callback(another_callback, "another_callback")
    assert called

    called = False
    sim_name = ""

    # Change target
    cudaq.set_target("stim")
    assert sim_name == "stim"
    assert called

    called = False
    sim_name = ""

    # Unregister
    cudaq.unregister_set_target_callback("another_callback")
    cudaq.set_target("density-matrix-cpu")
    assert sim_name == "density-matrix-cpu"
    assert not called

    # Reset target also triggers callback
    cudaq.reset_target()
    assert sim_name == "nvidia" or sim_name == "qpp-cpu"

    # Check other target info
    is_remote = None

    def yet_another_callback(target):
        nonlocal is_remote
        is_remote = target.is_remote()

    cudaq.set_target("quantinuum", emulated=True)
    cudaq.register_set_target_callback(yet_another_callback,
                                       "yet_another_callback")
    assert is_remote
    assert sim_name == "quantinuum"


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
