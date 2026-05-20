# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

install(FILES
  Synthesis/Gridsynth.h
  Synthesis/KmmSynthesize.h
  DESTINATION include/cudaq/Synthesis/Synthesis
)

install(FILES
  Circuit/Circuit.h
  Circuit/Gate.h
  DESTINATION include/cudaq/Synthesis/Circuit
)

install(FILES
  Math/Integer.h
  Math/Real.h
  Math/Types.h
  Math/Unitary.h
  DESTINATION include/cudaq/Synthesis/Math
)

install(FILES
  Math/Ring/Domega.h
  Math/Ring/Dsqrt2.h
  Math/Ring/Zomega.h
  Math/Ring/Zsqrt2.h
  DESTINATION include/cudaq/Synthesis/Math/Ring
)
