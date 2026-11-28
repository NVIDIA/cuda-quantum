# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(gadgets_core_test core.c)
target_link_libraries(gadgets_core_test PRIVATE cudaq::gadgets)
add_test(NAME gadgets_core COMMAND gadgets_core_test)
add_executable(gadgets_gadget_test gadget.cpp)
target_link_libraries(gadgets_gadget_test PRIVATE cudaq::gadgets)
target_compile_features(gadgets_gadget_test PRIVATE cxx_std_17)
add_test(NAME gadgets_gadget COMMAND gadgets_gadget_test)
add_executable(gadgets_circuit_test circuit.cpp)
target_link_libraries(gadgets_circuit_test PRIVATE cudaq::gadgets)
target_compile_features(gadgets_circuit_test PRIVATE cxx_std_17)
add_test(NAME gadgets_circuit COMMAND gadgets_circuit_test)
add_executable(gadgets_codes_test codes.cpp)
target_link_libraries(gadgets_codes_test PRIVATE cudaq::gadgets)
target_compile_features(gadgets_codes_test PRIVATE cxx_std_17)
add_test(NAME gadgets_codes COMMAND gadgets_codes_test)
