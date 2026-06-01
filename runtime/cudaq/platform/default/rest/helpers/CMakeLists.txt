# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors.  #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if (CUDAQ_ENABLE_ANYON_BACKEND)
  add_subdirectory(anyon)
endif()
if (AWSSDK_ROOT)
  add_subdirectory(braket)
endif()
add_subdirectory(infleqtion)
if (CUDAQ_ENABLE_IONQ_BACKEND)
  add_subdirectory(ionq)
endif()
if (CUDAQ_ENABLE_IQM_BACKEND)
  add_subdirectory(iqm)
endif()
if (CUDAQ_ENABLE_OQC_BACKEND)
  add_subdirectory(oqc)
endif()
if (CUDAQ_ENABLE_QCI_BACKEND)
  add_subdirectory(qci)
endif()
add_subdirectory(quantinuum)
if (CUDAQ_ENABLE_QUANTUM_MACHINES_BACKEND)
  add_subdirectory(quantum_machines)
endif()
if (CUDAQ_ENABLE_SCALEWAY_BACKEND)
  add_subdirectory(scaleway)
endif()
if(CUDAQ_ENABLE_TII_BACKEND)
  add_subdirectory(tii)
endif()
if(CUDAQ_ENABLE_QBRAID_BACKEND)
  add_subdirectory(qbraid)
endif()
