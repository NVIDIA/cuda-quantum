# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast

class MidCircuitMeasurementAnalyzer(ast.NodeVisitor):
    """The `MidCircuitMeasurementAnalyzer` is a utility class searches for 
       common measurement - conditional patterns to indicate to the runtime 
       that we have a circuit with mid-circuit measurement and subsequent conditional 
       quantum operation application."""

    def __init__(self):
        self.measureResultsVars = []
        self.hasMidCircuitMeasures = False

    def isMeasureCallOp(self, node):
        return isinstance(
            node, ast.Call) and node.__dict__['func'].id in ['mx', 'my', 'mz']

    def visit_Assign(self, node):
        target = node.targets[0]
        if not 'func' in node.value.__dict__:
            return
        creatorFunc = node.value.func
        if 'id' in creatorFunc.__dict__ and creatorFunc.id in [
                'mx', 'my', 'mz'
        ]:
            self.measureResultsVars.append(target.id)

    def visit_If(self, node):
        condition = node.test
        # catch if mz(q)
        if self.isMeasureCallOp(condition):
            self.hasMidCircuitMeasures = True 
            return 
        
        # Catch if val, where val = mz(q)
        if 'id' in condition.__dict__ and condition.id in self.measureResultsVars:
            self.hasMidCircuitMeasures = True
        # Catch if UnaryOp mz(q)
        elif isinstance(condition, ast.UnaryOp):
            self.hasMidCircuitMeasures = self.isMeasureCallOp(condition.operand)
        # Catch if something BoolOp mz(q)
        elif isinstance(condition,
                        ast.BoolOp) and 'values' in condition.__dict__:
            for node in condition.__dict__['values']:
                if self.isMeasureCallOp(node):
                    self.hasMidCircuitMeasures = True
                    break
