# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s


import pytest
import cudaq
from dataclasses import dataclass

def test_quantum_struct():
  @dataclass
  class patch:
      q : cudaq.qview
      r : cudaq.qview 

  @cudaq.kernel 
  def entry():
      q = cudaq.qvector(2)
      r = cudaq.qvector(2)
      p = patch(q, r)
      h(p.r[0])

  print(entry)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__entry()
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
# The struq type is erased in this example.
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

