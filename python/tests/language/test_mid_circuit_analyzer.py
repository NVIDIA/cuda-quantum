# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import ast 
import pytest
import numpy as np

import cudaq

def test_simple_0():
    '''Test we can find an if on an assigned bit variable.'''
    funcSrc = '''
def simple():
    q = cudaq.qvector(2)
    h(q[0])
    i = mz(q[0], "c0")
    if i:
        x(q[1])
    mz(q)
    '''

    module = ast.parse(funcSrc)
    analyzer = cudaq.MidCircuitMeasurementAnalyzer()
    analyzer.visit(module)
    assert analyzer.hasMidCircuitMeasures

def test_simple_1():
    '''Test we can find an if on a measure call operation.'''
    funcSrc = '''
def simple():
    q = cudaq.qvector(2)
    h(q[0])
    if mz(q[0], "c0"):
        x(q[1])
    mz(q)
    '''

    module = ast.parse(funcSrc)
    analyzer = cudaq.MidCircuitMeasurementAnalyzer()
    analyzer.visit(module)
    assert analyzer.hasMidCircuitMeasures

def test_simple_2():
    '''Test we can find an if with a unary operation like not on a measure call operation.'''
    
    funcSrc = '''
def simple():
    q = cudaq.qvector(2)
    h(q[0])
    if not mz(q[0], "c0"):
        x(q[1])
    mz(q)
    '''

    module = ast.parse(funcSrc)
    analyzer = cudaq.MidCircuitMeasurementAnalyzer()
    analyzer.visit(module)
    assert analyzer.hasMidCircuitMeasures


def test_simple_3():
    '''Test we can find an if with a binary boolean operation on measure operations.'''
    
    funcSrc = '''
def simple():
    q = cudaq.qvector(2)
    h(q[0])
    if mz(q[1]) and mz(q[0], "c0"):
        x(q[1])
    mz(q)
    '''

    module = ast.parse(funcSrc)
    analyzer = cudaq.MidCircuitMeasurementAnalyzer()
    analyzer.visit(module)
    assert analyzer.hasMidCircuitMeasures