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
  Synthesis/AgSynth.h
  DESTINATION include/cudaq/Synthesis/Synthesis
)

install(FILES
  Circuit/Circuit.h
  Circuit/Gate.h
  DESTINATION include/cudaq/Synthesis/Circuit
)

install(FILES
  Math/Real.h
  Math/Unitary.h
  DESTINATION include/cudaq/Synthesis/Math
)

install(FILES
  Support/Result.h
  DESTINATION include/cudaq/Synthesis/Support
)
