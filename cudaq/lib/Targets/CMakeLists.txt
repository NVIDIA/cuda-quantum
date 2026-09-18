# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Declarative target descriptions (<name>/<name>.yml), the source of truth
# for every in-tree CUDA-Q target. cudaq-target-db-gen reads them and emits
# the CUDAQTargetDatabase table. Generation is invoked from the top-level
# CMakeLists.txt after any CUDAQ_EXTERNAL_PROJECTS add_target_config()
# registrations have been processed.
