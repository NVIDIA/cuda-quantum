# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import sys

cudaq.set_target("quake_fake")


@cudaq.kernel
def early_return(return_early: bool) -> int:
    i = 0
    while i < 4:
        if return_early:
            return 1
        i += 1
    return i


@cudaq.kernel
def branch_returns(first_branch: bool) -> int:
    if first_branch:
        return 7
    else:
        return 9


@cudaq.kernel
def loop_control(skip: int, stop: int) -> int:
    i = 0
    total = 0
    while i < 6:
        if i == stop:
            break
        if i == skip:
            i += 1
            continue
        total += i
        i += 1
    return total


def check_result(kernel, expected, *args):
    results = cudaq.run(kernel, *args, shots_count=1)
    assert len(results) == 1
    assert results[0] == expected, f"expected {expected}, got {results[0]}"


def main():
    check_result(early_return, 1, True)
    check_result(early_return, 4, False)
    check_result(branch_returns, 7, True)
    check_result(branch_returns, 9, False)
    check_result(loop_control, 8, 2, 5)
    check_result(loop_control, 15, 9, 9)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
        sys.exit(1)
