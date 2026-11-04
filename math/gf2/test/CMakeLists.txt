# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(gf2_core_test core.c)
target_link_libraries(gf2_core_test PRIVATE cudaq::gf2)
add_test(NAME gf2_sparse_matrices COMMAND gf2_core_test)
add_executable(gf2_map_test typed_map.c)
target_link_libraries(gf2_map_test PRIVATE cudaq::gf2)
add_test(NAME gf2_typed_maps COMMAND gf2_map_test)
add_executable(gf2_owned_memory_test owned_memory.c)
target_link_libraries(gf2_owned_memory_test PRIVATE cudaq::gf2)
add_test(NAME gf2_owned_memory COMMAND gf2_owned_memory_test)
