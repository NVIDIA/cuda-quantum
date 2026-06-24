# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import time


class PerfTrace:
    """
    Utility class for recording and displaying performance tracing results.
    """
    __data = {}

    @staticmethod
    def add_trace(func_name, elapsed_time):
        if func_name not in PerfTrace.__data:
            PerfTrace.__data[func_name] = [elapsed_time]
        else:
            PerfTrace.__data[func_name].append(elapsed_time)

    @staticmethod
    def dump():
        # Sort by total time
        sorted_data = sorted(PerfTrace.__data.items(),
                             key=lambda x: sum(x[1]),
                             reverse=True)
        print("Performance tracing results:")
        print("Function name\tTotal time\tNumber of calls\tAverage time")
        print("-" * 50)
        for func_name, elapsed_times in sorted_data:
            print(
                f"{func_name}\t{sum(elapsed_times)}\t{len(elapsed_times)}\t{sum(elapsed_times) / len(elapsed_times):.6f}"
            )
            print()


class ScopeTimer(object):
    """
    Scope timer for performance tracing.
    Usage: specify a name for the scope and use it in a with statement:
    with ScopeTimer("evolve.integrator.integrate"):
        integrator.integrate(schedule.current_step)
    
    This will record the time taken by the specified scope and track it in the performance tracing results.
    """

    def __init__(self, name):
        self.__name = name

    def __enter__(self):
        self.__start = time.time()

    def __exit__(self, type, value, traceback):
        self.__finish = time.time()
        PerfTrace.add_trace(self.__name, self.__finish - self.__start)
