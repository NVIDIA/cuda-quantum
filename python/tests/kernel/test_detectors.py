# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest
import os


def test_basic():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        r = cudaq.qubit()
        mz(q)
        mz(r)
        detector(-2, -1)
        mz(q)
        mz(r)
        detector(-2, -1)

    dets = cudaq.detectors(mykernel)
    assert dets == [[0, 1], [2, 3]]


def test_detector_stdvec():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        r = cudaq.qubit()
        mz(q)
        mz(r)
        detector([-2, -1])
        mz(q)
        mz(r)
        detector([-2, -1])

    dets = cudaq.detectors(mykernel)
    assert dets == [[0, 1], [2, 3]]


def test_detector_stdvec_single_element():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        r = cudaq.qubit()
        mz(q)
        mz(r)
        detector([-2])

    dets = cudaq.detectors(mykernel)
    assert dets == [[0]]


def test_detector_loops():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        r = cudaq.qubit()
        mz(q)
        mz(r)
        for i in range(10):
            mz(q)
            mz(r)
            detector(-4, -2)
            detector(-3, -1)

    dets = cudaq.detectors(mykernel)
    assert len(dets) == 20
    for i in range(20):
        # 2 measurements per detector
        assert len(dets[i]) == 2
        # E.g. 0,2 / 1,3 / 4,6 / 5,7 / ...
        assert dets[i][0] == i
        assert dets[i][1] == i + 2


def test_detector_error_no_args():

    @cudaq.kernel
    def mykernel():
        detector()

    with pytest.raises(RuntimeError) as excinfo:
        cudaq.detectors(mykernel)
    assert "missing value" in str(excinfo.value)


def test_detector_subkernel():

    @cudaq.kernel
    def sub():
        detector(-1)

    @cudaq.kernel
    def main():
        q = cudaq.qubit()
        mz(q)
        sub()

    dets = cudaq.detectors(main)
    # mz(q) is index 0. detector(-1) refers to index 0.
    assert dets == [[0]]


def test_detector_positive_args():

    @cudaq.kernel
    def mykernel():
        q = cudaq.qubit()
        mz(q)
        detector(5)
        detector(10, -1)

    dets = cudaq.detectors(mykernel)
    # detector(5) -> [-5]
    # detector(10, -1) -> [-10, 0] (since -1 refers to index 0)
    assert dets == [[-5], [-10, 0]]


def test_detector_error_out_of_bounds():

    @cudaq.kernel
    def mykernel():
        detector(-1)

    # Should fail because there are no measurements
    with pytest.raises(RuntimeError) as excinfo:
        cudaq.detectors(mykernel)
    assert "Detector measurement index is negative" in str(excinfo.value)


def test_detector_dynamic_args():

    @cudaq.kernel
    def mykernel(i: int):
        q = cudaq.qubit()
        mz(q)
        detector(i)

    # detector(-1) -> index 0
    dets = cudaq.detectors(mykernel, -1)
    assert dets == [[0]]


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
