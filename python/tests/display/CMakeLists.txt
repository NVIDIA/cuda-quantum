# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
# # This source code and the accompanying materials are made available under #
# the terms of the Apache License 2.0 which accompanies this distribution. #
# ============================================================================ #

add_test(
  NAME cudaq-py-display-draw
  COMMAND ${PYTHON_EXECUTABLE} test_draw.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(
  cudaq-py-display-draw PROPERTIES ENVIRONMENT
                                    "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
